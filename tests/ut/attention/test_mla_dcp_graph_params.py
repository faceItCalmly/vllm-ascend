# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.common_cp import CPKVScope
from vllm_ascend.attention.context_parallel.mla_cp import (
    AscendMLADCPDecodeMetadata,
    MLADCPParamProvider,
    build_mla_dcp_fia_params,
)
from vllm_ascend.compilation.updatable_graph import GraphUpdateTask, SharedSource, UpdatableGraph
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import AscendAutoRegressiveSpeculator


def local_lens(lengths, rank, interleave):
    """Independent reference based on ownership of each token position."""
    return [sum((position // interleave) % 2 == rank for position in range(length)) for length in lengths]


def make_metadata(total=(13, 31, 0), query=(1, 1, 1), *, rank=1, interleave=1, causal=True):
    seq_lens = torch.tensor(total, dtype=torch.int32)
    decode = AscendMLADCPDecodeMetadata(
        input_positions=torch.zeros(sum(query), dtype=torch.int64),
        block_table=torch.arange(6).reshape(3, 2),
        seq_lens=seq_lens,
        max_seq_lens=max(total),
        seq_lens_list=list(total),
        actual_seq_lengths_q=list(accumulate(query)),
        cp_seq_len=local_lens(total, rank, interleave),
        cp_history_seq_len=local_lens([max(s - q, 0) for s, q in zip(total, query)], rank, interleave),
    )
    return SimpleNamespace(
        decode=decode,
        causal=causal,
        num_decode_tokens=sum(query),
        external_flashmla=None,
    )


def strict_fia_v2_out(*, actual_seq_kvlen, block_table, actual_seq_qlen=None):
    # SharedSource must produce kwargs accepted by FIA-v2, with no GQA aliases.
    return actual_seq_qlen, actual_seq_kvlen, block_table


def resolve_steps(steps, *, is_prefill=False, merged=False):
    """Model captured task identities and per-provider occurrence indices."""
    tasks = []
    counts = Counter()
    for step_index, step in enumerate(steps):
        for layer, metadata in step.items():
            split = metadata.causal and (is_prefill or (merged and step_index == 0))
            kinds = (CPKVScope.HISTORY, CPKVScope.CURRENT) if split else (CPKVScope.FULL,)
            for kind in kinds:
                provider = MLADCPParamProvider(layer, kind, pad_to=metadata.num_decode_tokens)
                # FULL keeps its captured query layout, including BSND lengths.
                query_ends = metadata.decode.actual_seq_lengths_q
                captured_qlen = [end - start for start, end in zip([0] + query_ends[:-1], query_ends)]
                tasks.append(
                    GraphUpdateTask(
                        strict_fia_v2_out,
                        {"actual_seq_qlen": captured_qlen},
                        provider,
                        counts[provider],
                        None,
                        None,
                    )
                )
                counts[provider] += 1
    return SimpleNamespace(tasks=tasks, provider_sizes=dict(counts))


@pytest.mark.parametrize("causal", [False, True])
def test_v1_mla_draft_parameters_bind_steps_and_accept_fia_v2_keywords(causal):
    steps = [
        {
            layer: make_metadata(total=(13 + step, 31 + step, 0), query=query, causal=causal)
            for layer in (("first", "second") if step % 2 == 0 else ("second", "first"))
        }
        for step, query in enumerate(((4, 2, 1), (1, 1, 1), (1, 1, 1)))
    ]
    proposer = SimpleNamespace(
        dcp_size=2,
        draft_model_config=SimpleNamespace(use_mla=True),
        _runnable=MagicMock(),
    )
    backend = object()
    with (
        patch("vllm_ascend.spec_decode.llm_base_proposer.use_updatable_graph", return_value=True),
        patch("vllm_ascend.spec_decode.llm_base_proposer.get_dcp_group") as get_group,
    ):
        AscendSpecDecodeBaseProposer._maybe_update_metadata(proposer, backend, steps)
    get_group.assert_not_called()
    proposer._runnable.set_attn_backend.assert_called_once_with(backend)
    params = proposer._runnable.update_draft_model_metadata.call_args.args[0]
    graph = resolve_steps(steps, merged=True)
    resolved = UpdatableGraph.resolve_tasks(graph, SharedSource(params))
    full_indices = Counter()
    for task in resolved:
        layer = task.provider.metadata_layer_name
        kind = task.provider.attention_kind
        if kind is CPKVScope.FULL:
            step_index = full_indices[layer] + int(causal)
            full_indices[layer] += 1
        else:
            step_index = 0
        metadata = steps[step_index][layer]
        qlen, kvlen, table = task.operation(**task.kwargs)
        if kind is CPKVScope.CURRENT:
            assert qlen == kvlen == metadata.decode.actual_seq_lengths_q
            assert table is None
        else:
            expected = metadata.decode.cp_history_seq_len if kind is CPKVScope.HISTORY else metadata.decode.cp_seq_len
            if kind is CPKVScope.FULL:
                expected = expected + [0] * max(metadata.num_decode_tokens - len(expected), 0)
            assert kvlen == expected
            assert table is metadata.decode.block_table


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("interleave", [1, 8])
def test_v2_mla_draft_parameters_preserve_layout_lengths_and_padding(is_prefill, causal, rank, interleave):
    query = (4, 2, 1) if is_prefill else (1, 1, 1)
    steps = [
        {
            layer: make_metadata(total=(14 + step, 32, 0), query=query, rank=rank, interleave=interleave, causal=causal)
            for layer in ("first", "second")
        }
        for step in range(1 if is_prefill else 2)
    ]
    for step_index, step in enumerate(steps):
        for layer, metadata in step.items():
            if step_index:
                metadata.decode.block_table = steps[0][layer].decode.block_table
            if not is_prefill:
                # Subsequent draft lengths must be computed from the input batch.
                metadata.decode.cp_seq_len = [-1, -1, -1]
    speculator = object.__new__(AscendAutoRegressiveSpeculator)
    speculator.use_dcp = True
    speculator.attn_architecture = "MLA"
    speculator.dcp_manager = SimpleNamespace(dcp_world_rank=rank)
    speculator.draft_vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, cp_kv_cache_interleave_size=interleave)
    )
    speculator.input_batch = SimpleNamespace(num_reqs=2, seq_lens_np=[13, 31])
    speculator.num_speculative_steps = 3
    speculator.max_model_len = 32
    speculator.draft_attn_layer_names = set(steps[0])
    graph = resolve_steps(steps, is_prefill=is_prefill)
    params = speculator.build_fia_params(3, steps[0], is_prefill)
    resolved = UpdatableGraph.resolve_tasks(graph, SharedSource(params))
    for task in resolved:
        metadata = steps[task.provider_index][task.provider.metadata_layer_name]
        kind = task.provider.attention_kind
        qlen, kvlen, table = task.operation(**task.kwargs)
        if kind is CPKVScope.CURRENT:
            assert qlen == kvlen == metadata.decode.actual_seq_lengths_q
            assert table is None
        else:
            totals = (14 + task.provider_index, 32, 0)
            if kind is CPKVScope.HISTORY:
                totals = [max(s - q, 0) for s, q in zip(totals, query)]
                assert qlen == metadata.decode.actual_seq_lengths_q
            else:
                assert qlen == list(query)
            expected = local_lens(totals, rank, interleave)
            if kind is CPKVScope.FULL and is_prefill:
                expected += [0] * (metadata.num_decode_tokens - len(expected))
            assert kvlen == expected
            assert table is metadata.decode.block_table
        if not is_prefill:
            assert metadata.decode.cp_seq_len == [-1, -1, -1]


@pytest.mark.parametrize("external_flashmla", [None, object()])
def test_mla_publisher_skips_layers_without_fia_tasks(external_flashmla):
    metadata = make_metadata()
    metadata.external_flashmla = external_flashmla
    if external_flashmla is None:
        metadata.decode = None
    assert build_mla_dcp_fia_params("layer", metadata, is_draft_model_prefill=True) == []


def test_padding_size_does_not_split_one_draft_provider_identity():
    assert MLADCPParamProvider("layer", CPKVScope.FULL, pad_to=7) == MLADCPParamProvider(
        "layer", CPKVScope.FULL, pad_to=3
    )
