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
"""A3 W8 cache lifecycle, checked against an independent CPU FP16 golden.

Graphs are serialized on one explicit capture stream. Memory checks are small
bounded leak checks with allocator slack, not production memory-savings claims.
"""

import gc
import json

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def run_regression(result):
    torch.npu.set_device(0)
    op = torch.vllm_ascendC.grouped_matmul_situ_quant
    result.update(device=str(torch.npu.get_device_name(0)), torch=torch.__version__)

    def layer(seed, k=256, n=512):
        gen = torch.Generator().manual_seed(seed)
        logical = [torch.randint(-8, 8, (k, n), generator=gen, dtype=torch.int8) for _ in range(2)]
        scales = [torch.rand(n, generator=gen) * 0.001 + 0.0001 for _ in range(2)]
        packed = []
        for w in logical:
            lanes = w.reshape(k, -1, 8).to(torch.int64)
            p = torch.zeros(lanes.shape[:-1], dtype=torch.int64)
            for i in range(8):
                p |= (lanes[..., i] & 15) << (4 * i)
            packed.append(p.to(torch.int32).npu())
        encoded = [(s.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).npu() for s in scales]
        return logical, scales, packed, encoded

    def inputs(c=32, k=256):
        gen = torch.Generator().manual_seed(91 + c + k)
        x = torch.randint(-32, 32, (c, k), generator=gen, dtype=torch.int8)
        xs = torch.rand(c, generator=gen) * 0.01 + 0.001
        return x, xs, x.npu(), xs.npu()

    def golden(inp, weights, counts):
        ys, ss, acts = [], [], []
        offset = 0
        for w, ws, rows in zip(weights[0], weights[1], counts):
            if rows:
                acc = inp[0][offset : offset + rows].float() @ w.float()
                hidden = (acc * ws).half().float() * inp[1][offset : offset + rows, None]
                g, u = hidden.chunk(2, dim=-1)
                gate = (2 * torch.sigmoid(g / 2) - 1) * torch.sigmoid(g) * 4
                act = gate * ((2 * torch.sigmoid(u * (2 / 25)) - 1) * 25)
                maximum = act.abs().amax(dim=-1)
                scale = maximum / 127
                inv = torch.where(maximum > 0, 127 / maximum, torch.zeros_like(maximum))
                ys.append((act * inv[:, None]).round().clamp(-128, 127).to(torch.int8))
                ss.append(scale)
                acts.append(act)
            offset += rows
        return torch.cat(ys), torch.cat(ss), torch.cat(acts)

    def invoke(inp, weights, groups):
        return op(inp[2], weights[2], weights[3], inp[3], groups, [], 4.0, 25.0, 1)

    def check(out, expected, label):
        torch.npu.synchronize()
        y, s = [v.cpu() for v in out]
        ry, rs, act = expected
        assert y.shape == ry.shape and s.shape == rs.shape, label
        assert y.dtype == torch.int8 and s.dtype == torch.float32, label
        diff = (y.to(torch.int16) - ry.to(torch.int16)).abs()
        row = dict(
            scenario=label,
            int8_max_diff=int(diff.max()),
            int8_one_step_fraction=float((diff == 1).float().mean()),
            scale_max_abs_diff=float((s - rs).abs().max()),
        )
        result["checks"].append(row)
        assert torch.allclose(s, rs, rtol=1e-3, atol=1e-5), row
        assert row["int8_max_diff"] <= 1 and row["int8_one_step_fraction"] < 0.02, row
        assert torch.allclose(y.float() * s[:, None], act, rtol=2e-2, atol=2e-2), row

    def passed(label):
        result["scenarios"].append(label)
        print("PASS: " + label, flush=True)

    a, b = layer(1001), layer(200003)
    inp = inputs()
    counts = [32, 0]
    groups = torch.tensor(counts, dtype=torch.int64).npu()
    refs = [golden(inp, w, counts) for w in (a, b)]
    assert not torch.equal(refs[0][0], refs[1][0]), "A and B must be distinguishable"
    for index in (0, 0, 0, 1, 0):
        check(invoke(inp, (a, b)[index], groups), refs[index], f"eager {'AB'[index]}")
    passed("eager A/A and A/B/A with CPU golden")

    # Default-stream eager calls above warm metadata. There is intentionally no
    # operator warmup on this new stream before its first GLOBAL capture.
    stream = torch.npu.Stream()
    torch.npu.synchronize()
    graphs, outputs = [], []
    for w in (a, b):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
            out = invoke(inp, w, groups)
        graphs.append(graph)
        outputs.append(out)
    with torch.npu.stream(stream):
        for index in (0, 0, 0, 0, 1, 0):
            graphs[index].replay()
            check(outputs[index], refs[index], f"graph {'AB'[index]}")
        graphs[0].replay()
        check(outputs[0], refs[0], "graph A before eager B")
        check(invoke(inp, b, groups), refs[1], "eager B between A replays")
        graphs[0].replay()
        check(outputs[0], refs[0], "graph A after eager B")
    passed("fresh stream capture; serial A repeats; graph A/B/A; graph A/eager B/graph A")

    # C growth affects accumulation scratch only. K growth separately forces a
    # larger actual W8 allocation on the same stream. An explicit frozen-W8
    # guard is valid only if old graphs remain usable after that rejection.
    for label, big_inp, big_layer, big_counts in (
        ("C-only scratch growth", inputs(1024), a, [512, 512]),
        ("actual W8 growth K=512", inputs(32, 512), layer(9999, k=512), [16, 16]),
    ):
        big_groups = torch.tensor(big_counts, dtype=torch.int64).npu()
        torch.npu.synchronize()
        with torch.npu.stream(stream):
            try:
                grown_out = invoke(big_inp, big_layer, big_groups)
            except RuntimeError as exc:
                guard = "grouped_matmul_situ_quant: graph-visible W8 capacity is frozen"
                if label != "actual W8 growth K=512" or guard not in str(exc):
                    raise
                result["w8_growth"] = dict(status="explicitly_rejected", error=str(exc))
                torch.npu.synchronize()
            else:
                check(grown_out, golden(big_inp, big_layer, big_counts), label)
                del grown_out
                if label == "actual W8 growth K=512":
                    result["w8_growth"] = dict(status="supported_and_golden_checked")
            for index in (0, 1, 0):
                graphs[index].replay()
                check(outputs[index], refs[index], label + f" old graph {'AB'[index]}")
        passed(label + " then old graph A/B/A")

    with torch.npu.stream(stream):
        for current in ([0, 32], [32, 0], [0, 32]):
            groups.copy_(torch.tensor(current, dtype=torch.int64))
            graphs[0].replay()
            check(outputs[0], golden(inp, a, current), "dynamic captured groups")
            replacement = torch.tensor(current, dtype=torch.int64).npu()
            check(invoke(inp, a, replacement), golden(inp, a, current), "new group tensor same W8")
        # Adapter casts must execute on replay and observe changed source data.
        for dtype in (torch.int32, torch.float32):
            source = torch.tensor([32, 0], dtype=dtype).npu()
            check(invoke(inp, a, source), refs[0], "adapter eager " + str(dtype))
            adapter = torch.npu.NPUGraph()
            with torch.npu.graph(adapter, stream=stream, capture_error_mode="global"):
                adapter_out = invoke(inp, a, source)
            for current in ([0, 32], [32, 0]):
                source.copy_(torch.tensor(current, dtype=dtype))
                adapter.replay()
                check(adapter_out, golden(inp, a, current), "adapter dynamic " + str(dtype))
            del adapter_out, adapter
    passed("dynamic zero-expert groups via in-place update/new tensor and int32/float32 adapters")

    expected = golden(inp, a, [0, 32])
    pending = []
    with torch.npu.stream(stream):
        for _ in range(16):
            x, xs, gl = inp[2].clone(), inp[3].clone(), groups.clone()
            disposable = (inp[0], inp[1], x, xs)
            pending.append(invoke(disposable, a, gl))
            for tensor in (x, xs, gl):
                tensor.set_(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
        for out in pending:
            check(out, expected, "immediate input set_ disposal")
    del pending, out
    passed("asynchronous submission followed by immediate x/xscale/groups set_ disposal")

    def allocated():
        torch.npu.synchronize()
        gc.collect()
        torch.npu.synchronize()
        return int(torch.npu.memory_allocated())

    eager_samples = [allocated()]
    with torch.npu.stream(stream):
        for batch in range(6):
            for index in (0, 1) * 16:
                out = invoke(inp, (a, b)[index], groups)
            check(out, golden(inp, b, [0, 32]), f"memory eager batch {batch}")
            del out
            eager_samples.append(allocated())
    # Retain original graphs to ensure fixed-state replay stays valid. Repeated
    # temporary captures release their graph/output refs before each sample;
    # graph allocator retention is allowed modest slack, not exact-zero growth.
    capture_samples = [allocated()]
    for batch in range(8):
        temporary = torch.npu.NPUGraph()
        with torch.npu.graph(temporary, stream=stream, capture_error_mode="global"):
            temporary_out = invoke(inp, a, groups)
        with torch.npu.stream(stream):
            temporary.replay()
            check(temporary_out, expected, f"released capture {batch}")
        del temporary_out, temporary
        capture_samples.append(allocated())
    result["memory"] = dict(
        eager_allocated_bytes=eager_samples,
        released_capture_allocated_bytes=capture_samples,
        eager_late_growth_bytes=eager_samples[-1] - eager_samples[2],
        capture_late_growth_bytes=capture_samples[-1] - capture_samples[2],
        eager_slack_bytes=8 * 2**20,
        capture_slack_bytes=32 * 2**20,
        scope="Small synthetic bounded leak check; allocator retention allowed; not production memory savings",
    )
    assert result["memory"]["eager_late_growth_bytes"] <= 8 * 2**20, result["memory"]
    assert result["memory"]["capture_late_growth_bytes"] <= 32 * 2**20, result["memory"]
    with torch.npu.stream(stream):
        for index in (0, 1, 0):
            graphs[index].replay()
            check(outputs[index], golden(inp, (a, b)[index], [0, 32]), "retained graph after memory loop")
    passed("bounded eager/released-capture allocation growth and retained graph replay")
    result["status"] = "passed"


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Requires A3 GMSQ extension")
def test_a3_w8_cache_lifecycle():
    # Load the A3 binding only after the device skip, so other builds collect.
    import vllm_ascend.vllm_ascendC  # noqa: F401

    result = dict(status="running", checks=[], scenarios=[])
    run_regression(result)
    print(json.dumps(result["memory"], indent=2))
