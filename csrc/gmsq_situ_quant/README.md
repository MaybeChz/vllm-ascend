# A3 GMSQ cache and graph contract

The direct `torch.vllm_ascendC.grouped_matmul_situ_quant` operator fuses
GMM1, SiTU and per-token INT8 quantization. Its expanded W8 cache covers
all local experts; routing counts remain dynamic on every call.

## Static metadata and dynamic inputs

Immutable metadata is keyed by device, every weight/scale address and format,
K, N and group-list type. Capacity C and group-list contents are not cache keys.
Warm up every static configuration eagerly before graph capture. A metadata
miss during capture fails before format conversion or CPU-to-device copying.
Retained tensor references prevent address reuse from aliasing an old entry.

Weights and scales must remain immutable after warmup. In-place weight updates
are not detected; use new tensor storage and warm it up before capturing a new
graph. Existing graphs continue to reference their original configuration.

The operator accepts one-dimensional group lists with at least E entries in
INT64, INT32 or FP32. It normalizes them to contiguous INT64 on the device,
including inside a graph. Counts must already represent valid nonnegative
integers or cumulative offsets; conversion does not validate those values.
The caller remains responsible for valid totals within the input capacity.

## One W8 slot per device and stream

Each slot owns its W8 allocation, owner ID and decision storage. Every call
enqueues three kernels on the same stream:

1. `gmsq_w8_cache_prepare` compares the current owner with the configuration ID
   and publishes one stable decision for all fused-kernel cores. A miss clears
   the owner before overwriting W8.
2. `gmsq_fused_256_cached` skips unpacking on a hit. On a miss it unpacks all
   experts before running the existing GEMM and quantization pipeline.
3. `gmsq_w8_cache_publish` publishes ownership after the fused kernel completes.

All three kernels are captured and replayed. Ownership is checked at execution
time, not remembered by the host at capture time. A/A can hit; A/B/A must
rebuild W8 for both switches. This also covers an eager B call inserted between
two replays of graph A on the same stream.

All operations sharing a slot must execute serially. Concurrent replay of
graphs captured against the same slot is unsupported, even if callers launch
those graphs from different streams. The protocol is not device-error recovery.

Owner and decision occupy separate 64-byte cache lines. AIV control accesses
use aligned DMA and explicit pipeline events. AIC invalidates its decision
cache line before its single scalar read, so all cores take the same conditional
synchronization path.

## Capture initialization and allocation lifetime

Eager warmup initializes a device reserve of 64 control pairs (8 KiB) before
publishing it. A fresh capture stream claims an initialized pair without
recording an owner-reset operation in the graph. Capture fails explicitly if
the reserve is exhausted; eager warmup can replenish it. Control banks remain
alive because their allocation stream can differ from their consuming stream.
Eager initialization must not overlap another global capture on the device.

Before capture, the W8 slot can grow to the largest warmed configuration on its
device. Once a graph uses that slot, its capacity is frozen. A larger request
fails before replacing the allocation, and existing graphs remain usable.
Use a new stream with suitable eager warmup for larger configurations.

Accumulator growth for a different C does not change W8 ownership. Old
graph-visible accumulator allocations remain alive for old graphs. Metadata,
control banks and retained scratch are not automatically reclaimed when a graph
is destroyed; new configurations or streams can increase memory usage.

The asynchronous host callback snapshots raw addresses and retains their
`c10::Storage` objects, so input `Tensor.set_()` immediately after submission
cannot release the storage still needed by the queued call. Accumulator
workspace starts at offset zero; it does not reserve a second W8-sized prefix.

## Regression tests

On A3 with the rebuilt extension installed:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 TASK_QUEUE_ENABLE=1 python -m pytest -sv \
  tests/e2e/pull_request/one_card/aclgraph/test_gmsq_w8_cache_lifecycle.py \
  tests/e2e/pull_request/one_card/aclgraph/test_gmsq_group_list.py
```

The lifecycle test uses an independent CPU FP16-scaled golden for ND weights.
It covers A/A, A/B/A, fresh-stream capture, graph/eager interleaving, accumulator
growth, rejected W8 growth followed by old-graph replay, immediate input disposal
and bounded repeated-capture allocation checks. The group-list test compares
INT32/FP32 and strided inputs against the contiguous INT64 kernel reference,
including dynamic replay, empty capacity and invalid metadata.

These tests do not establish NZ accuracy, arbitrary concurrent replay safety,
real-model token quality or end-to-end serving throughput. Performance comparisons
must include prepare, fused and publish rather than reporting only the fused
kernel duration.
