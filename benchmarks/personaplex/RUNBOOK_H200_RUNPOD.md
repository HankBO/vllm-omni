# PersonaPlex H200 RunPod Test Runbook

This runbook compares Stage 0 depformer CUDA graphs off/on on one H200. Mimi encoder CUDA graphs remain enabled throughout. It covers:

- Unprofiled Realtime sweeps at 1, 2, 4, 8, 16, 24, and 64 sessions, with depformer graphs off and on.
- The depformer CUDA-event benchmark at the same batch sizes.
- Paired Nsight depformer profiles at n=16, with depformer graphs off and on.

Nsight traces are diagnostic. Do not report or compare client RTF from profiled runs: profiling perturbs pacing. Use `--capture-range-end=stop` so capture completion does not shut down the server.

## 1. Connect to the Pod

Create a RunPod H200 pod with a TCP-enabled SSH endpoint. In RunPod's Connect panel, note the public host and mapped SSH port. Run inference and benchmarks on the Pod; transfer the input WAV and finished artifacts over SCP.

On the local workstation:

```bash
export POD_HOST='<RunPod public host>'
export POD_SSH_PORT='<mapped SSH port>'
ssh -p "$POD_SSH_PORT" "root@$POD_HOST"
```

All commands below run in the Pod shell unless marked local. The API binds to Pod localhost, so it does not need a public mapped API port. To access it from the local workstation instead of running the benchmark driver on the Pod, open a tunnel in a separate local terminal:

```bash
ssh -N -L 8099:127.0.0.1:8099 -p "$POD_SSH_PORT" "root@$POD_HOST"
```

## 2. Pin Code, Environment, and Inputs

Use this tested source revision for every run:

```bash
git clone https://github.com/vllm-project/vllm-omni.git /workspace/vllm-omni
cd /workspace/vllm-omni
git checkout --detach be5bab061bc8aae41a2326ddd8c4d0c64ca5deeb
python3 --version  # Use Python 3.12.
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.31.0 --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/db9527a46873454610df6dbedf79a36d6bf1a7f6
uv pip install -e .
```

These commands follow the repository's CUDA setup for this source revision. The RunPod image must provide a compatible NVIDIA driver; vLLM's default wheel here is CUDA 12.9. Keep vLLM, PyTorch, CUDA, and vLLM-Omni versions fixed throughout the comparison; do not independently upgrade packages between variants. If using a prepared environment instead, activate it and verify it matches these requirements.

```bash
python - <<'PY'
import torch
import vllm
import vllm_omni

print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("vllm_omni", vllm_omni.__file__)
PY
nvidia-smi
```

Create the remote input directory on the Pod, then transfer a known 24 kHz assistant WAV from the local workstation:

```bash
# Pod shell
mkdir -p /workspace/vllm-omni/tmp
```

```bash
# Local workstation; adjust the local input path as needed.
scp -P "$POD_SSH_PORT" /path/to/input_assistant.wav \
  "root@$POD_HOST:/workspace/vllm-omni/tmp/input_assistant.wav"
```

On the Pod, download the model into a stable local directory. If Hub authentication is required, run `hf auth login` interactively; never place a token in a command or this runbook.

```bash
export HF_HOME=/workspace/hf_cache
export MODEL=/workspace/models/personaplex-7b-v1
mkdir -p "$(dirname "$MODEL")"
hf download nvidia/personaplex-7b-v1 --local-dir "$MODEL"
test -f "$MODEL/model.safetensors"
test -f "$MODEL/voices.tgz"
test -f "$MODEL/tokenizer_spm_32k_3.model"
test -f tmp/input_assistant.wav
python - <<'PY'
import wave
with wave.open("tmp/input_assistant.wav") as wav:
    print("input:", wav.getframerate(), "Hz", wav.getnchannels(), "channels", wav.getnframes(), "samples")
PY
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

## 3. Create a 64-Session H200 Deploy Config

The checked-in deploy config defaults to 24 sessions. Generate a temporary H200 config rather than editing the tracked YAML. For a 141 GiB-class H200, use Stage 0 GPU utilization 0.75 and Stage 1 0.15. Set the duplex limit and both stages' `max_num_seqs` to 64. The test input is 375 frames (30 seconds), not a full 3000-frame KV window; verify startup cache capacity and admission logs before testing 64 sessions.

```bash
cd /workspace/vllm-omni
export DEPLOY_CONFIG=/workspace/vllm-omni/tmp/personaplex_h200_64.yaml
python - <<'PY'
import yaml
from pathlib import Path

source = Path("vllm_omni/deploy/personaplex.yaml")
target = Path("tmp/personaplex_h200_64.yaml")
config = yaml.safe_load(source.read_text())
config["duplex_session"]["max_sessions"] = 64
for stage in config["stages"]:
    stage["max_num_seqs"] = 64
    if stage["stage_id"] == 0:
        stage["gpu_memory_utilization"] = 0.75
        stage["hf_overrides"]["mimi_cuda_graphs"] = True
        stage["hf_overrides"]["depformer_cuda_graphs"] = True
    elif stage["stage_id"] == 1:
        stage["gpu_memory_utilization"] = 0.15
        stage["hf_overrides"]["mimi_cuda_graphs"] = True
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(config, sort_keys=False))
print(target)
PY
```

Check startup logs for Stage 0 KV capacity, depformer graph capture through padded batch 64, and 64 Mimi stream slots. If the KV cache cannot admit 64 live requests, do not interpret an n=64 rejection as a graph performance result; adjust the H200 memory budget/config first and record the change.

## 4. Run Unprofiled Realtime Sweeps

Run two server configurations, changing only Stage 0 depformer graphs. Stage 0 Mimi encoder graphs and Stage 1 Mimi decoder graphs remain on in both. Run the server in one Pod terminal and the sweep in a second Pod terminal. Stop each server with Ctrl-C after its sweep.

In both Pod terminals, set up the environment:

```bash
cd /workspace/vllm-omni
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache
export MODEL=/workspace/models/personaplex-7b-v1
export DEPLOY_CONFIG=/workspace/vllm-omni/tmp/personaplex_h200_64.yaml
export SWEEP_ROOT=/workspace/vllm-omni/tmp/personaplex_h200_sweeps_be5bab061
```

### Depformer Graphs Off

Server terminal:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME="$HF_HOME" \
python -m vllm_omni.entrypoints.cli.main serve "$MODEL" \
  --omni --deploy-config "$DEPLOY_CONFIG" \
  --stage-overrides '{"0":{"hf_overrides":{"depformer_cuda_graphs":false}}}' \
  --served-model-name nvidia/personaplex-7b-v1 \
  --host 127.0.0.1 --port 8099
```

Sweep terminal:

```bash
python benchmarks/personaplex/realtime_sweep.py run \
  --label dep-off_mimi-on --depformer-graphs off --mimi-graphs on \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model nvidia/personaplex-7b-v1 \
  --input-wav tmp/input_assistant.wav \
  --sessions 1 2 4 8 16 24 64 --repeats 3 \
  --load-frames 375 --drain-s 5 \
  --root "$SWEEP_ROOT" \
  --server-revision be5bab061bc8aae41a2326ddd8c4d0c64ca5deeb \
  --server-hardware 'NVIDIA H200'
```

Do not pass `--max-client-rtf`: collect every point without an RTF gate while retaining RTF measurements for separate evaluation. Then stop the server gracefully with Ctrl-C.

### Depformer Graphs On

Restart the server with the same command, changing only its stage override:

```bash
--stage-overrides '{"0":{"hf_overrides":{"depformer_cuda_graphs":true}}}'
```

Run the second sweep with a distinct label and the same output root:

```bash
python benchmarks/personaplex/realtime_sweep.py run \
  --label dep-on_mimi-on --depformer-graphs on --mimi-graphs on \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model nvidia/personaplex-7b-v1 \
  --input-wav tmp/input_assistant.wav \
  --sessions 1 2 4 8 16 24 64 --repeats 3 \
  --load-frames 375 --drain-s 5 \
  --root "$SWEEP_ROOT" \
  --server-revision be5bab061bc8aae41a2326ddd8c4d0c64ca5deeb \
  --server-hardware 'NVIDIA H200'
```

Summarize the unprofiled matrix:

```bash
python benchmarks/personaplex/realtime_sweep.py summarize \
  --root "$SWEEP_ROOT" --default-mimi on \
  --output-file "$SWEEP_ROOT/summary.md"
```

The sweep refuses to overwrite an existing `load-result.json`. Use a fresh output root for a rerun or deliberately move old results aside.

## 5. Run the CUDA-Event Depformer Benchmark

Run after stopping the server. This standalone benchmark uses random weights and measures the depformer only; it is not a whole-service result.

```bash
python benchmarks/personaplex/depformer_cuda_event_bench.py \
  --batch-sizes 1 2 4 8 16 24 64 \
  --iters 300 --warmup 30 \
  --output-file "$SWEEP_ROOT/depformer_cuda_events.md"
```

It checks eager/graph output equality and verifies every timed graph call replayed. A failed graph capture, equality assertion, or replay check invalidates the corresponding result.

## 6. Capture Nsight Depformer Profiles at n=16

Keep Mimi encoder graphs on in both profiles. Compare only depformer graphs off vs on, using n=16 and 375 frames. Do not apply an RTF criterion or report client RTF from these profiled runs. The driver still writes an RTF field in raw JSON; it is instrumentation-contaminated and must be excluded from Nsight comparisons.

The server starts CUDA profiling at runner tick 20 and stops at tick 120. `--cuda-graph-trace=node` is required to see kernels replayed inside CUDA graphs. Use the same RunPod terminal setup from the sweep section in both server and sweep terminals.

Confirm Nsight Systems is installed in the Pod image before starting:

```bash
nsys --version
```

### Eager Depformer Profile

Server terminal:

```bash
mkdir -p "$SWEEP_ROOT"
CUDA_VISIBLE_DEVICES=0 HF_HOME="$HF_HOME" \
nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node --trace-fork-before-exec=true --force-overwrite=true \
  -o "$SWEEP_ROOT/eager_n16" \
  python -m vllm_omni.entrypoints.cli.main serve "$MODEL" \
    --omni --deploy-config "$DEPLOY_CONFIG" \
    --stage-overrides '{"0":{"hf_overrides":{"depformer_cuda_graphs":false}}}' \
    --served-model-name nvidia/personaplex-7b-v1 \
    --host 127.0.0.1 --port 8099
```

Sweep terminal:

```bash
python benchmarks/personaplex/realtime_sweep.py run \
  --label eager --depformer-graphs off --mimi-graphs on \
  --url 'ws://127.0.0.1:8099/v1/realtime?duplex=1' \
  --model nvidia/personaplex-7b-v1 --input-wav tmp/input_assistant.wav \
  --sessions 16 --load-frames 375 --drain-s 5 \
  --root /workspace/vllm-omni/tmp/personaplex_h200_nsys_load_be5bab061 \
  --server-revision be5bab061bc8aae41a2326ddd8c4d0c64ca5deeb \
  --server-hardware 'NVIDIA H200'
```

The profiled sweep may miss frame-deficit/voicing acceptance because profiling slows the server. Preserve its JSON as diagnostic data, do not report RTF, and stop the server gracefully with Ctrl-C after the client finishes so the `.nsys-rep` finalizes.

### Graphed Depformer Profile

Repeat the same procedure, changing the Nsight output to `"$SWEEP_ROOT/graphs_n16"` and the stage override to:

```bash
--stage-overrides '{"0":{"hf_overrides":{"depformer_cuda_graphs":true}}}'
```

Keep the Nsight options, Mimi graph setting, input, session count, and drain period unchanged. Analyze the pair:

```bash
python analyze_nsys.py "$SWEEP_ROOT" \
  --output-file "$SWEEP_ROOT/nsys_depformer_n16.md"
```

The report compares depformer-correlated GPU-kernel envelope median/p99, kernels per call, host launch calls, and depformer-envelope idle. This idle percentage is not global GPU idle. Exclude client RTF from the Nsight report.

## 7. Transfer Artifacts Over SCP

From the local workstation after the Pod runs finish:

```bash
scp -P "$POD_SSH_PORT" -r \
  "root@$POD_HOST:/workspace/vllm-omni/tmp/personaplex_h200_sweeps_be5bab061" \
  ./tmp/
scp -P "$POD_SSH_PORT" -r \
  "root@$POD_HOST:/workspace/vllm-omni/tmp/personaplex_h200_nsys_load_be5bab061" \
  ./tmp/
```

The `.nsys-rep` files can be large. Copy SQLite exports and Markdown reports for routine review; retain the original `.nsys-rep` files for deeper inspection.

## 8. Record and Interpret Results

For each unprofiled point, record requested/passed sessions, client RTF median/worst, first-audio latency, output-frame deficit, voiced frames, and pacing warnings. A driver PASS without an RTF ceiling means functional acceptance, not necessarily real-time throughput; state any RTF threshold explicitly in final claims.

For Nsight, record graph mode, n=16, capture range, depformer call count, GPU wall median/p99, kernel count, host launch count, and depformer-envelope idle. Exclude client RTF and do not treat profiled load-test pass counts as production capacity. Pair eager/graph traces under the same H200, config, input, and capture procedure.

At completion, verify the expected artifacts:

```bash
find "$SWEEP_ROOT" -name load-result.json | sort
find "$SWEEP_ROOT" \( -name '*.nsys-rep' -o -name '*.sqlite' \) -print | sort
```

Expected unprofiled points: 7 session counts x 3 repeats for each of the two graph modes, or 42 results total. Expected Nsight profiles: eager and graphs at n=16. Report the exact commit, RunPod GPU SKU, CUDA/driver, vLLM/vLLM-Omni versions, deploy settings, and any RTF threshold used.