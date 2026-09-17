# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for Personaplex depformer static KV and CUDA-graph wrapper control flow."""

from __future__ import annotations

import pytest
import torch

from tests.model_executor.models.personaplex.duplex._depformer_testing import (
    clone_depformer,
    frame,
    make_depformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_depformer_cudagraph import (
    CUDAGraphDepformerWrapper,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_padding_rows_do_not_affect_live_rows() -> None:
    live = make_depformer(seed=3)
    padded = clone_depformer(live)
    text1, hidden1, tokens1, provided1 = frame(batch=1, seed=4)
    text2, hidden2, tokens2, provided2 = frame(batch=2, seed=5)
    text2[0] = text1[0]
    hidden2[0] = hidden1[0]
    tokens2[0] = tokens1[0]
    provided2[0] = provided1[0]

    out1 = live(text1, hidden1, audio_tokens=tokens1, audio_provided=provided1)
    out2 = padded(text2, hidden2, audio_tokens=tokens2, audio_provided=provided2)
    torch.testing.assert_close(out2[0], out1[0], rtol=0, atol=0)


def test_kv_reset_isolates_successive_frames() -> None:
    streaming = make_depformer(seed=6)
    isolated = clone_depformer(streaming)
    first = frame(batch=1, seed=7)
    second = frame(batch=1, seed=8)
    streaming(*first[:2], audio_tokens=first[2], audio_provided=first[3])
    streamed_second = streaming(*second[:2], audio_tokens=second[2], audio_provided=second[3])
    only_second = isolated(*second[:2], audio_tokens=second[2], audio_provided=second[3])
    torch.testing.assert_close(streamed_second, only_second, rtol=0, atol=0)


def test_wrapper_selects_smallest_capture_size_that_fits() -> None:
    model = make_depformer()
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1, 2, 4, 8], enabled=False)
    assert wrapper._select_padded_b(1) == 1
    assert wrapper._select_padded_b(3) == 4
    assert wrapper._select_padded_b(9) is None


def test_disabled_wrapper_stays_eager() -> None:
    model = make_depformer(seed=9)
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1, 2], enabled=False)
    wrapper.warmup(torch.device("cpu"))
    text, hidden, tokens, provided = frame(batch=1, seed=10)
    out_wrap = wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
    out_eager = model(text, hidden, audio_tokens=tokens, audio_provided=provided)
    torch.testing.assert_close(out_wrap, out_eager, rtol=0, atol=0)
    stats = wrapper.stats_snapshot()
    assert stats["calls"] == 1
    assert stats["eager"] == 1
    assert stats["replays"] == 0
    assert stats["num_graphs"] == 0


def test_cpu_warmup_does_not_capture() -> None:
    model = make_depformer()
    wrapper = CUDAGraphDepformerWrapper(model, capture_sizes=[1, 2], enabled=True)
    wrapper.warmup(torch.device("cpu"))
    assert not wrapper.is_ready
    text, hidden, tokens, provided = frame(batch=2, seed=11)
    wrapper(text, hidden, audio_tokens=tokens, audio_provided=provided)
    assert wrapper.stats.eager == 1
    assert wrapper.stats.replays == 0
