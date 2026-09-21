#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""A3 GMSQ group-list normalization, capture, and input-lifetime regressions.

Compare valid rows with the contiguous-int64 kernel reference. Independent
kernel accuracy is covered by test_gmsq_w8_cache_lifecycle.py.
"""

import gc
import re
from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

DTYPES = (torch.int64, torch.int32, torch.float32)


def pack_int4(logical):
    """Pack eight signed INT4 lanes into each INT32 value."""
    grouped = logical.reshape(logical.shape[0], -1, 8).to(torch.int64)
    packed = torch.zeros(grouped.shape[:-1], dtype=torch.int64)
    for i in range(8):
        packed |= (grouped[..., i] & 15) << (4 * i)
    return packed.to(torch.int32)


@dataclass
class Case:
    experts: int
    capacity: int
    x: torch.Tensor
    x_scale: torch.Tensor
    weights: list
    scales: list


def make_case(experts, capacity, k=256, n=512):
    weights, scales = [], []
    for _ in range(experts):
        weights.append(pack_int4(torch.randint(-8, 8, (k, n))).npu())
        scale = torch.rand(n, dtype=torch.float32) * 0.02 + 0.001
        scales.append((scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).npu())
    return Case(
        experts,
        capacity,
        torch.randint(-32, 32, (capacity, k), dtype=torch.int8).npu(),
        (torch.rand(capacity) * 0.01 + 0.001).npu(),
        weights,
        scales,
    )


def counts_for(case, changed=False):
    c = case.capacity
    if case.experts == 1:
        return [c // 2 if changed else c]
    return [0, c // 2] if changed else [c // 3, c - c // 3]


def encoded_counts(counts, gl_type, dtype):
    values = torch.tensor(counts, dtype=torch.int64)
    if gl_type == 0:
        values = values.cumsum(0)
    return values.to(dtype)


def make_groups(counts, gl_type, dtype, strided=False, extra=False):
    values = encoded_counts(counts, gl_type, dtype)
    if extra:
        # The kernel reads only the first E entries. A harmless extra entry
        # verifies the promised numel >= E contract.
        values = torch.cat((values, values.new_zeros(1)))
    if strided:
        backing = torch.zeros(values.numel() * 2, device="npu", dtype=dtype)
        result = backing[::2]
        result.copy_(values)
        assert result.numel() > 1 and not result.is_contiguous()
        return result
    return values.npu()


def invoke(case, groups, gl_type):
    return torch.vllm_ascendC.grouped_matmul_situ_quant(
        case.x, case.weights, case.scales, case.x_scale, groups, [], 1.0, None, gl_type
    )


def to_host(outputs, valid_rows):
    torch.npu.synchronize()
    return tuple(t[:valid_rows].cpu() for t in outputs)


def check(outputs, expected, valid_rows, label):
    actual = to_host(outputs, valid_rows)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=label)


def reference(case, counts, gl_type):
    groups = make_groups(counts, gl_type, torch.int64)
    return to_host(invoke(case, groups, gl_type), sum(counts))


def check_empty_capacity(case):
    empty_case = Case(case.experts, 0, case.x[:0], case.x_scale[:0], case.weights, case.scales)
    for gl_type in (0, 1):
        for dtype in DTYPES:
            groups = make_groups([0] * case.experts, gl_type, dtype)
            y, ys = invoke(empty_case, groups, gl_type)
            assert y.shape == (0, case.weights[0].shape[1] * 4), y.shape
            assert ys.shape == (0,), ys.shape
            assert y.dtype == torch.int8 and ys.dtype == torch.float32
            assert y.device == case.x.device and ys.device == case.x.device
    torch.npu.synchronize()
    print("PASS C=0: three dtypes, both group_list types, empty outputs", flush=True)
    # Invalid metadata must still fail before the C=0 fast return.
    check_host_guards(empty_case)


def check_host_guards(case):
    # Reject malformed metadata on the host before launching the kernel.
    valid = counts_for(case)
    bad_inputs = [
        ("bool", make_groups(valid, 1, torch.bool), r"dtype|int64|int32|float32|type"),
        ("float16", make_groups(valid, 1, torch.float16), r"dtype|int64|int32|float32|type"),
        ("double", make_groups(valid, 1, torch.float64), r"dtype|int64|int32|float32|type"),
        ("0D", torch.tensor(case.capacity, device="npu", dtype=torch.int64), r"1\s*[- ]?d|one.dim|dim|rank"),
        ("2D", make_groups(valid, 1, torch.int64).reshape(1, case.experts), r"1\s*[- ]?d|one.dim|dim|rank"),
        ("short", make_groups(valid, 1, torch.int64)[: case.experts - 1], r"numel|length|size|expert|entries|elements"),
        ("CPU", torch.tensor(valid, dtype=torch.int64), r"device|npu"),
    ]
    for name, groups, detail_pattern in bad_inputs:
        try:
            invoke(case, groups, 1)
        except RuntimeError as exc:
            message = str(exc)
            assert "group_list" in message.lower(), (name, message)
            assert re.search(detail_pattern, message, re.IGNORECASE), (name, message)
            # Device failures do not count as clear host validation errors.
            assert not re.search(r"107030|507\d{3}|SUSPECT REMOTE|DEVICE ERROR", message, re.IGNORECASE), (
                name,
                message,
            )
        else:
            raise AssertionError(f"Host did not reject {name} group_list")
    torch.npu.synchronize()
    print(f"PASS host guards: E={case.experts}, seven invalid metadata cases", flush=True)


def check_configuration(case, gl_type, dtype, strided):
    label = f"E={case.experts} C={case.capacity} glType={gl_type} dtype={dtype} strided={strided}"
    counts = counts_for(case)
    changed_counts = counts_for(case, changed=True)
    expected = reference(case, counts, gl_type)
    changed_expected = reference(case, changed_counts, gl_type)
    groups = make_groups(counts, gl_type, dtype, strided)

    # Warm every dtype conversion and metadata configuration before GLOBAL
    # capture; its default side stream is intentionally not supplied explicitly.
    for _ in range(2):
        check(invoke(case, groups, gl_type), expected, sum(counts), label + " eager")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        outputs = invoke(case, groups, gl_type)
    graph.replay()
    check(outputs, expected, sum(counts), label + " captured")

    # Both changed and original groups are replayed: casting/contiguous copies
    # must remain graph nodes and observe source contents on every replay.
    for current, want in ((changed_counts, changed_expected), (counts, expected), (changed_counts, changed_expected)):
        groups.copy_(encoded_counts(current, gl_type, dtype))
        torch.npu.synchronize()
        graph.replay()
        check(outputs, want, sum(current), label + " dynamic replay")

    pending = []
    for _ in range(8):
        original = make_groups(counts, gl_type, dtype, strided)
        result = invoke(case, original, gl_type)
        # No synchronize between submission and disposal. For an int64
        # contiguous input this also checks the no-conversion aliasing path.
        original.set_(torch.empty(0, device=original.device, dtype=original.dtype))
        pending.append(result)
        # Neither the original group tensor nor its normalized int64 temporary
        # is retained by this test between calls. Subsequent conversions churn
        # small allocator blocks while the prior task may still be queued.
        pending.append(invoke(case, make_groups(counts, gl_type, dtype, strided), gl_type))
    for result in pending:
        check(result, expected, sum(counts), label + " temporary/disposed input")
    del pending, outputs, graph
    gc.collect()
    print("PASS " + label, flush=True)


def run_regression():
    torch.npu.set_device(0)

    torch.manual_seed(20260921)
    cases = [make_case(1, 32), make_case(2, 32), make_case(1, 4096)]
    # Check metadata errors before the valid capture configurations.
    for case in cases[:2]:
        check_host_guards(case)
        counts = counts_for(case)
        ids = torch.tensor([e for e, count in enumerate(counts) for _ in range(count)], dtype=torch.int32).npu()
        histogram = torch.histc(ids, bins=case.experts, min=0, max=case.experts)
        groups = histogram.unsqueeze(0).sum(dim=0)
        expected = reference(case, counts, 1)
        check(invoke(case, groups, 1), expected, sum(counts), "histc counts")
        print(
            f"PASS histc->sum->GMSQ E={case.experts} histc_dtype={histogram.dtype} group_dtype={groups.dtype}",
            flush=True,
        )
    check_empty_capacity(cases[1])
    configurations = 0
    for case in cases:
        for gl_type in (0, 1):
            for dtype in DTYPES:
                for strided in (False, True) if case.experts == 2 else (False,):
                    check_configuration(case, gl_type, dtype, strided)
                    configurations += 1

    case = cases[1]
    counts = counts_for(case)
    for gl_type in (0, 1):
        expected = reference(case, counts, gl_type)
        for dtype in DTYPES:
            check(
                invoke(case, make_groups(counts, gl_type, dtype, extra=True), gl_type),
                expected,
                sum(counts),
                f"extra entry {dtype} glType={gl_type}",
            )
    print("PASS six numel > E cases", flush=True)
    print(
        f"PASS {configurations} normalization/capture/lifetime configurations; "
        "21 host rejection cases; six C=0 cases; six extra-entry cases",
        flush=True,
    )


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires A3 GMSQ extension")
def test_a3_gmsq_group_list_normalization():
    # Load the A3 binding only after the device skip, so other builds collect.
    import vllm_ascend.vllm_ascendC  # noqa: F401

    run_regression()
