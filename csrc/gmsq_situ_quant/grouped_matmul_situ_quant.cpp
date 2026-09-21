/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// GroupedMatmulSituQuant op_host.
// Fused GMM1 (fake-A8W8) + SiTU + per-token INT8 quant, single operator.
//
// DEVICE-SIDE group_list parsing (this revision):
//   The fused single-launch path performs NO host-side group_list parse. The
//   group_list device tensor is normalized to contiguous int64 on-device, then
//   parsed by the kernel every forward (mE / startRow / mPadOff / sumMPad / MValid are
//   derived in-kernel from GM). This makes the operator correct for DYNAMIC
//   production MoE group_list (new tensor per forward OR in-place value update)
//   without host-side value copies or synchronization.
//
//   Persistent metadata entries are keyed by device, full weight/scale pointer
//   and format sets, K, N and group_list_type. Neither capacity C nor group_list
//   identity/values are part of this key. Every layer must be warmed up eagerly:
//   metadata misses during graph capture are rejected before conversion/H2D.
//
//   Mutable scratch is separate for each device/stream. The framework may
//   allocate scratch during capture, including on its internal side stream.
//   Old allocations remain alive after growth for previously captured graphs.
//   A device prepass decides whether the concrete W8 allocation owns this
//   static configuration; a postpass publishes ownership after fused completion.
//   Both execute on every replay. Graphs sharing a
//   capture stream's scratch must replay serially; concurrent replay is not
//   supported by this reuse scheme. Weights/scales remain static after warmup.
//
//   All mechanisms preserved: column-blocked w8, fused 3-stage pipeline,
//   NZ (FRACTAL_NZ) input, stacked/TensorList, capacity-padded x, gl_type 0/1.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/core/Storage.h>

#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

#include "acl/acl_rt.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "grouped_matmul_situ_quant.h"

namespace vllm_ascend {

// Device kernel launcher exported by libvllm_ascend_kernels (ccec-compiled TU
// owns the kernel entry and the tripple-chevron launch; see op_kernel/).
void gmsq_fused_256_impl(uint32_t blockDim, void *stream, void *x, void *wPtrTbl, void *scPtrTbl,
                          void *w8, void *acc, void *scaleF32, void *xScale, void *y, void *yScale,
                          void *groupList, int32_t E, int32_t kNum, int32_t nNum, int32_t nBlock,
                          int32_t NP, int32_t N, int32_t N2, int32_t K, int32_t C, int32_t glType,
                          float beta, float invBeta, int32_t hasLinear, float linBeta,
                          float invLinBeta, int32_t skipUnpack, int32_t nzInput);

// Additive ABI: the legacy export above remains available for archived hosts.
void gmsq_fused_256_cached_impl(uint32_t blockDim, void *stream, void *x, void *wPtrTbl, void *scPtrTbl,
                              void *w8, void *acc, void *scaleF32, void *xScale, void *y, void *yScale,
                              void *groupList, int32_t E, int32_t kNum, int32_t nNum, int32_t nBlock,
                              int32_t NP, int32_t N, int32_t N2, int32_t K, int32_t C, int32_t glType,
                              float beta, float invBeta, int32_t hasLinear, float linBeta,
                              float invLinBeta, int32_t nzInput, void *owner, void *decision,
                              uint64_t configId);

// AIV epilogue UB budget: the x_scale chunk buffer + static per-row buffers
// must stay within the AIV UB. Measured overflow threshold on Ascend910_9382
// is between C=16384 (pass) and C=18432 (UB OOB, error 507015). Guard at the
// conservative bound so callers can catch and fall back instead of crashing
// the device.
constexpr int64_t GMSQ_MAX_CAPACITY = 16384;

constexpr int32_t GMSQ_BM = 128;
constexpr int32_t GMSQ_BN = 128;
constexpr int32_t GMSQ_BK = 64;
constexpr int32_t GMSQ_EPI_M = 32;

// ACL tensor format ids (see acl_base.h / torch_npu Format enum).
constexpr int64_t ACL_FMT_ND = 2;
constexpr int64_t ACL_FMT_FRACTAL_NZ = 29;

// ---------------------------------------------------------------------------
// Persistent device-side metadata cache for the FUSED single-launch path.
// Key = static weight configuration ONLY (weights are static during inference):
//   fullWPtrs/fullSPtrs — data_ptr of every expert's weight / weight_scale
//   device             — owning NPU
//   weight/scale format sets and K, N — static tensor layout
//   glType             — group_list interpretation mode (0 cumsum / 1 count)
// group_list identity is deliberately NOT in the key (it changes every forward
// in production MoE; the kernel parses it on-device each forward).
// ---------------------------------------------------------------------------
struct FusedMetaCache {
    uint64_t configId = 0;  // 0 is invalid; IDs never wrap or get reused.
    // key
    int64_t device = -1;
    std::vector<uintptr_t> fullWPtrs;
    std::vector<uintptr_t> fullSPtrs;
    std::vector<int64_t> weightFormats;
    std::vector<int64_t> scaleFormats;
    int64_t K = 0;
    int64_t N = 0;
    int64_t glType = 0;
    // NZ -> ND one-time converted tensors (all E, kept alive).
    bool srcIsNz = false;
    std::vector<at::Tensor> ndWeights;
    std::vector<at::Tensor> ndScales;
    // Strong refs to ORIGINAL weight/weight_scale tensors (all E) so the
    // allocator cannot reuse their data_ptr while this cache is alive.
    std::vector<at::Tensor> weightRefs;
    std::vector<at::Tensor> scaleRefs;
    // device tensors (static per weight config)
    at::Tensor tbl;     // int32 [4*E] = wPtrV(2E) + scPtrV(2E)
    int64_t scOff = 0;
    int32_t kNum = 0;
    int32_t nNum256 = 0;
    int32_t gemmNBlock = 1024;  // P54: MTE2 load-balance N-block, computed ONCE per
                                // cache build (removed from steady-state launch path)
    int64_t wBytes = 0;   // E*K*N
};

// These three tensors always move together. A control pair is never reused for
// another W8 allocation, even when an uncaptured eager allocation is replaced.
struct FusedW8Allocation {
    at::Tensor data;
    at::Tensor owner;
    at::Tensor decision;
};

struct FusedControlReserve {
    int64_t device = -1;
    at::Tensor freeBlock;
    // A block can be created on one eager stream and consumed by another.
    // Retain old 8KiB blocks so callback destruction after enqueue cannot free
    // their allocator storage on the wrong stream while device work is pending.
    std::vector<at::Tensor> retiredBlocks;
    int64_t next = 0;
};

constexpr int64_t GMSQ_CONTROL_RESERVE_PAIRS = 64;
constexpr int64_t GMSQ_CONTROL_LINE_WORDS = 16;  // 64B, one scalar DCache line
constexpr int64_t GMSQ_CONTROL_PAIR_WORDS = 2 * GMSQ_CONTROL_LINE_WORDS;

struct FusedScratchCache {
    int64_t device = -1;
    aclrtStream stream = nullptr;
    at::Tensor wsFlat;
    FusedW8Allocation w8;
    bool wsCaptured = false;
    bool w8Captured = false;
    // A captured custom kernel retains raw addresses, not these Tensor refs.
    // Keep replaced graph-visible accumulator allocations until shutdown.
    // W8 capacity freezes at first capture, so it has no retired allocations.
    std::vector<at::Tensor> retired;
};

static std::mutex g_fusedCacheMutex;
static std::vector<std::unique_ptr<FusedMetaCache>> g_fusedMetaEntries;
static std::vector<std::unique_ptr<FusedScratchCache>> g_fusedScratchEntries;
static std::vector<std::unique_ptr<FusedControlReserve>> g_fusedControlReserves;
static uint64_t g_nextFusedConfigId = 1;

// Called with g_fusedCacheMutex held. Initialization completes eagerly before a
// control pair is published. Capturing zeros here would reset owner on EVERY
// graph replay. Uninitialized device memory cannot safely serve as an owner.
static FusedControlReserve &GetControlReserve(int64_t device, const at::TensorOptions &options,
                                              bool capturing)
{
    FusedControlReserve *reserve = nullptr;
    for (const auto &entry : g_fusedControlReserves) {
        if (entry->device == device) {
            reserve = entry.get();
            break;
        }
    }
    if (reserve == nullptr) {
        TORCH_CHECK(!capturing,
                    "grouped_matmul_situ_quant: control reserve missing during graph capture; "
                    "perform eager warmup on this device first");
        auto entry = std::make_unique<FusedControlReserve>();
        entry->device = device;
        reserve = entry.get();
        g_fusedControlReserves.push_back(std::move(entry));
    }
    if (!reserve->freeBlock.defined() || reserve->next == GMSQ_CONTROL_RESERVE_PAIRS) {
        TORCH_CHECK(!capturing,
                    "grouped_matmul_situ_quant: initialized W8 control reserve exhausted during "
                    "graph capture; perform another eager warmup before capture");
        const int64_t words = GMSQ_CONTROL_RESERVE_PAIRS * GMSQ_CONTROL_PAIR_WORDS;
        at::Tensor replacement = at::empty({words}, options);
        TORCH_CHECK(reinterpret_cast<uintptr_t>(replacement.data_ptr()) % 64 == 0,
                    "grouped_matmul_situ_quant: W8 control storage must be 64-byte aligned");
        at::Tensor zerosCpu = at::zeros({words}, at::TensorOptions().dtype(at::kInt));
        replacement.copy_(zerosCpu, /*non_blocking=*/false);
        if (reserve->freeBlock.defined()) {
            reserve->retiredBlocks.push_back(reserve->freeBlock);
        }
        reserve->freeBlock = replacement;
        reserve->next = 0;
    }
    return *reserve;
}

// Return an ND tensor for `t` (original if already ND, else FRACTAL_NZ -> ND).
static at::Tensor EnsureNd(const at::Tensor &t, std::vector<at::Tensor> &converted)
{
    if (at_npu::native::get_npu_format(t) == ACL_FMT_ND) {
        return t;
    }
    at::Tensor nd = at_npu::native::npu_format_cast(t, ACL_FMT_ND);
    converted.push_back(nd);
    return nd;
}

static std::vector<int32_t> BuildPtrVec(const std::vector<uintptr_t> &ptrs)
{
    std::vector<int32_t> v;
    v.reserve(ptrs.size() * 2);
    for (uintptr_t p : ptrs) {
        v.push_back(static_cast<int32_t>(static_cast<uint32_t>(p & 0xFFFFFFFFu)));
        v.push_back(static_cast<int32_t>(p >> 32));
    }
    return v;
}

std::tuple<at::Tensor, at::Tensor> grouped_matmul_situ_quant(
    const at::Tensor &x, at::TensorList weight, at::TensorList weight_scale,
    const at::Tensor &x_scale, const at::Tensor &group_list, at::TensorList weight_assist_matrix,
    double beta, std::optional<double> linear_beta, int64_t group_list_type)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(x.scalar_type() == at::kChar, "x must be int8");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "x_scale must be float32");
    TORCH_CHECK(!weight.empty(), "weight list must be non-empty");
    TORCH_CHECK(weight.size() == weight_scale.size(), "weight/weight_scale size mismatch");
    TORCH_CHECK(weight_assist_matrix.empty() || weight_assist_matrix.size() == weight.size(),
                "weight_assist_matrix size mismatch");
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1, "group_list_type must be 0 or 1");
    TORCH_CHECK(group_list.dim() == 1, "group_list must be 1D");
    TORCH_CHECK(group_list.numel() >= static_cast<int64_t>(weight.size()),
                "group_list numel must be at least the number of experts");
    TORCH_CHECK(group_list.scalar_type() == at::kLong || group_list.scalar_type() == at::kInt ||
                    group_list.scalar_type() == at::kFloat,
                "group_list dtype must be int64, int32 or float32");

    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on an NPU device");
    const int64_t device = x.get_device();
    TORCH_CHECK(c10_npu::current_device() == device,
                "grouped_matmul_situ_quant: current NPU device must match x.device");
    TORCH_CHECK(x_scale.device() == x.device() && group_list.device() == x.device(),
                "x_scale and group_list must be on x.device");
    for (size_t e = 0; e < weight.size(); ++e) {
        TORCH_CHECK(weight[e].device() == x.device() && weight_scale[e].device() == x.device(),
                    "weight and weight_scale must be on x.device (expert ", e, ")");
    }
    for (const auto &assist : weight_assist_matrix) {
        TORCH_CHECK(assist.device() == x.device(), "weight_assist_matrix must be on x.device");
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t aivCoreNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAiv());
    int32_t aicCoreNum = static_cast<int32_t>(ascendcPlatform->GetCoreNumAic());
    TORCH_CHECK(aivCoreNum > 0 && aicCoreNum > 0, "failed to query core numbers");

    int64_t C = x.sizes()[0];      // x rows (capacity, >= M_valid)
    int64_t K = x.sizes()[1];
    int64_t experts = static_cast<int64_t>(weight.size());
    int64_t NP = weight[0].sizes()[1];  // packed N/8 per row
    int64_t N = NP * 8;
    int64_t N2 = N / 2;
    TORCH_CHECK(K % GMSQ_BK == 0, "K must be a multiple of 64");
    TORCH_CHECK(N % GMSQ_BN == 0, "N must be a multiple of 128");
    TORCH_CHECK(N2 % GMSQ_BN == 0, "N/2 must be a multiple of 128");
    TORCH_CHECK(weight[0].scalar_type() == at::kInt, "weight must be int32 packed");
    TORCH_CHECK(weight_scale[0].scalar_type() == at::kLong, "weight_scale must be int64 carrier");

    auto devOptI8 = at::TensorOptions().device(x.device()).dtype(at::kChar);
    auto devOptI32 = at::TensorOptions().device(x.device()).dtype(at::kInt);
    auto devOptF32 = at::TensorOptions().device(x.device()).dtype(at::kFloat);

    // outputs: y [C, N2] int8, y_scale [C] fp32 (only first M_valid rows meaningful)
    at::Tensor y = at::empty({C, N2}, devOptI8);
    at::Tensor y_scale = at::empty({C}, devOptF32);

    // EP empty rank (M_valid == 0): x may be [0, K] with an all-zero group_list
    // (short requests under TP8+EP leave some ranks without routed tokens). All
    // parameter validation is done above and the outputs are already empty with
    // the right shapes — return without building caches, reading group_list or
    // launching the kernel. A C > 0 input whose device-side group_list is all
    // zero stays on the normal launch path; the kernel tolerates
    // sum(group_list) == 0 (the P54 waveBatch clamp keeps batches at 0, so no
    // tile is scheduled and no cross-core flag pair is exercised).
    if (C == 0) {
        return {y, y_scale};
    }

    TORCH_CHECK(C <= GMSQ_MAX_CAPACITY,
                "grouped_matmul_situ_quant: capacity C=", C,
                " exceeds the supported maximum ", GMSQ_MAX_CAPACITY,
                " (AIV UB budget); reduce max_num_batched_tokens for this op");

    // AllToAll histograms may arrive as float32. Normalize on the input device
    // and current stream on EVERY call: counts change between graph replays.
    // Contiguous int64 input is reused without a copy. Float inputs must carry
    // finite, nonnegative integer counts (or valid cumulative counts for type 0),
    // with total valid rows <= C; value validation is the caller's contract.
    // Do not read values on the CPU or cache this converted tensor.
    at::Tensor kernelGroupList = group_list;
    if (kernelGroupList.scalar_type() != at::kLong) {
        kernelGroupList = kernelGroupList.to(at::kLong);
    }
    kernelGroupList = kernelGroupList.contiguous();

    float betaF = static_cast<float>(beta);
    float invBeta = 1.0f / betaF;
    int32_t hasLinear = linear_beta.has_value() ? 1 : 0;
    float lbF = hasLinear ? static_cast<float>(*linear_beta) : 1.0f;
    float invLb = hasLinear ? 1.0f / lbF : 1.0f;

    // ---- FUSED single-launch path (device-side group_list parse) ----
    int32_t nNum256 = static_cast<int32_t>(N / 256);
    bool fusedEligible = (N % 256 == 0) && (nNum256 <= 128) && (experts > 0);
    if (fusedEligible) {
        const aclrtStream gmsqStream = c10_npu::getCurrentNPUStream().stream();
        aclmdlRICaptureStatus captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        aclmdlRI captureModel = nullptr;
        const aclError captureError = aclmdlRICaptureGetInfo(gmsqStream, &captureStatus, &captureModel);
        TORCH_CHECK(captureError == ACL_SUCCESS,
                    "grouped_matmul_situ_quant: capture status query failed: ", captureError);
        TORCH_CHECK(captureStatus != ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED,
                    "grouped_matmul_situ_quant: cannot enqueue on an invalidated graph capture");
        const bool capturing = captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
        // Cheap host-side key (no H2D/D2H, no sync): full pointer sets + shapes +
        // gl_type. group_list is deliberately NOT in the key.
        std::vector<uintptr_t> fullWPtrs(static_cast<size_t>(experts));
        std::vector<uintptr_t> fullSPtrs(static_cast<size_t>(experts));
        std::vector<int64_t> weightFormats(static_cast<size_t>(experts));
        std::vector<int64_t> scaleFormats(static_cast<size_t>(experts));
        int64_t nzCount = 0;
        for (int64_t e = 0; e < experts; e++) {
            TORCH_CHECK(weight[e].data_ptr() != nullptr && weight_scale[e].data_ptr() != nullptr,
                        "weight and weight_scale must have non-null storage");
            fullWPtrs[static_cast<size_t>(e)] = reinterpret_cast<uintptr_t>(weight[e].data_ptr());
            fullSPtrs[static_cast<size_t>(e)] = reinterpret_cast<uintptr_t>(weight_scale[e].data_ptr());
            int64_t wfmt = at_npu::native::get_npu_format(weight[e]);
            weightFormats[static_cast<size_t>(e)] = wfmt;
            scaleFormats[static_cast<size_t>(e)] = at_npu::native::get_npu_format(weight_scale[e]);
            nzCount += (wfmt == ACL_FMT_FRACTAL_NZ) ? 1 : 0;
        }
        TORCH_CHECK(nzCount == 0 || nzCount == experts,
                    "grouped_matmul_situ_quant: mixed ND/FRACTAL_NZ weight experts are not "
                    "supported; all experts must use the same format");

        // Debug/test affordance: force the native-NZ unpack path regardless of
        // the reported tensor format. Needed on torch_npu builds whose
        // npu_format_cast cannot materialize a 2D-int32 FRACTAL_NZ (the bytes
        // must then already hold the int8-NZ tiling of the packed weight).
        static const bool forceNzInput = (std::getenv("GMSQ_FORCE_NZ_INPUT") != nullptr);
        bool srcIsNz = forceNzInput || (nzCount == experts);
        // Entries never move or expire; the mutex protects construction and
        // scratch growth. Published metadata is immutable and shared across
        // eager and capture streams on the same device.
        std::unique_lock<std::mutex> cacheLock(g_fusedCacheMutex);
        FusedMetaCache *meta = nullptr;
        for (const auto &entry : g_fusedMetaEntries) {
            if (entry->device == device && entry->K == K && entry->N == N &&
                entry->glType == group_list_type && entry->srcIsNz == srcIsNz &&
                entry->fullWPtrs == fullWPtrs && entry->fullSPtrs == fullSPtrs &&
                entry->weightFormats == weightFormats && entry->scaleFormats == scaleFormats) {
                meta = entry.get();
                break;
            }
        }

        if (meta == nullptr) {
            TORCH_CHECK(!capturing,
                        "grouped_matmul_situ_quant: metadata cache miss during graph capture; "
                        "warm up every weight/scale configuration eagerly on this device first "
                        "(pointer-table H2D and format conversion are forbidden during capture)");
            // Construct privately so a failed build cannot publish partial data.
            auto newEntry = std::make_unique<FusedMetaCache>();
            FusedMetaCache &fc = *newEntry;
            TORCH_CHECK(g_nextFusedConfigId != std::numeric_limits<uint64_t>::max(),
                        "grouped_matmul_situ_quant: W8 configuration ID space exhausted");
            fc.configId = g_nextFusedConfigId++;
            fc.device = device;
            fc.srcIsNz = srcIsNz;
            // pointer tables for ALL experts (expert index = table index)
            std::vector<uintptr_t> wPtrsAll;
            std::vector<uintptr_t> scPtrsAll;
            wPtrsAll.reserve(static_cast<size_t>(experts));
            scPtrsAll.reserve(static_cast<size_t>(experts));
            for (int64_t e = 0; e < experts; e++) {
                at::Tensor wNd;
                if (srcIsNz) {
                    // Native int4 FRACTAL_NZ: the int32 carrier views an int8
                    // packed-NZ storage (W4A8 convention: process_weights
                    // transpose -> maybe_trans_nz -> view(int32)). Keep the
                    // native geometry — no TransData, the kernel unpacks from
                    // int8-NZ addressing directly.
                    wNd = weight[static_cast<size_t>(e)];
                } else {
                    wNd = EnsureNd(weight[static_cast<size_t>(e)], fc.ndWeights);
                    TORCH_CHECK(wNd.is_contiguous(), "weight[e] must be contiguous (ND)");
                }
                at::Tensor sNd = EnsureNd(weight_scale[static_cast<size_t>(e)], fc.ndScales);
                fc.weightRefs.push_back(weight[static_cast<size_t>(e)]);
                fc.scaleRefs.push_back(weight_scale[static_cast<size_t>(e)]);
                TORCH_CHECK(sNd.is_contiguous(), "weight_scale[e] must be contiguous (ND)");
                wPtrsAll.push_back(reinterpret_cast<uintptr_t>(wNd.data_ptr()));
                scPtrsAll.push_back(reinterpret_cast<uintptr_t>(sNd.data_ptr()));
            }
            std::vector<int32_t> wPtrV = BuildPtrVec(wPtrsAll);
            std::vector<int32_t> scPtrV = BuildPtrVec(scPtrsAll);
            int64_t wPtrLen = static_cast<int64_t>(wPtrV.size());
            int64_t scOff = wPtrLen;
            int64_t tblLen = wPtrLen + static_cast<int64_t>(scPtrV.size());
            std::vector<int32_t> tblV;
            tblV.reserve(static_cast<size_t>(tblLen));
            tblV.insert(tblV.end(), wPtrV.begin(), wPtrV.end());
            tblV.insert(tblV.end(), scPtrV.begin(), scPtrV.end());
            at::Tensor tblCpu = at::from_blob(
                tblV.data(), {tblLen}, [](void *) {}, at::TensorOptions().dtype(at::kInt));
            at::Tensor tbl = at::empty({tblLen}, devOptI32);
            tbl.copy_(tblCpu, /*non_blocking=*/false);  // H2D — one-time per weight config

            fc.fullWPtrs = std::move(fullWPtrs);
            fc.fullSPtrs = std::move(fullSPtrs);
            fc.weightFormats = std::move(weightFormats);
            fc.scaleFormats = std::move(scaleFormats);
            fc.K = K; fc.N = N; fc.glType = group_list_type;
            fc.tbl = tbl;
            fc.scOff = scOff;
            fc.kNum = static_cast<int32_t>(K / GMSQ_BK);
            fc.nNum256 = nNum256;
            fc.wBytes = experts * K * N;

            // P54: MTE2 load-balance N-block scheduling computed ONCE per weight
            // config (cache build), removing the per-call scalar scheduling loop
            // from the steady-state launch path. Same first-principles rule as the
            // original per-call loop: critB = waves*blk minimized, largest blk on
            // ties to minimize A re-fetch / per-tile scalar overhead.
            {
                int32_t cores = aicCoreNum;
                int64_t bestCritB = INT64_MAX;
                int32_t gblock = 1024;
                for (int32_t blk = 1024; blk >= 256; blk -= 256) {
                    if (N % blk != 0) {
                        continue;  // clean tiling only (N is a multiple of 256)
                    }
                    int64_t tiles = experts * (N / blk);
                    int32_t waves = static_cast<int32_t>((tiles + cores - 1) / cores);
                    int64_t critB = static_cast<int64_t>(waves) * blk;
                    if (critB < bestCritB) {
                        bestCritB = critB;
                        gblock = blk;
                    }
                }
                fc.gemmNBlock = gblock;
            }
            meta = newEntry.get();
            g_fusedMetaEntries.push_back(std::move(newEntry));
        }

        const FusedMetaCache &fc = *meta;
        FusedScratchCache *scratch = nullptr;
        for (const auto &entry : g_fusedScratchEntries) {
            if (entry->device == device && entry->stream == gmsqStream) {
                scratch = entry.get();
                break;
            }
        }
        if (scratch == nullptr) {
            auto entry = std::make_unique<FusedScratchCache>();
            entry->device = device;
            entry->stream = gmsqStream;
            scratch = entry.get();
            g_fusedScratchEntries.push_back(std::move(entry));
        }

        // Freeze W8 once a graph records its address. Reject BEFORE changing
        // either scratch allocation, so old graphs survive a rejected request.
        const bool requestExceedsW8 = !scratch->w8.data.defined() || scratch->w8.data.numel() < fc.wBytes;
        TORCH_CHECK(!requestExceedsW8 || !scratch->w8Captured,
                    "grouped_matmul_situ_quant: graph-visible W8 capacity is frozen; warm up all "
                    "larger weight configurations before capture, or use a new stream");
        int64_t w8Capacity = fc.wBytes;
        if (!scratch->w8Captured) {
            // Includes a previously eager-used stream's FIRST capture, even
            // when other layers were warmed on a different stream.
            for (const auto &entry : g_fusedMetaEntries) {
                if (entry->device == device) {
                    w8Capacity = std::max(w8Capacity, entry->wBytes);
                }
            }
        }
        const bool growW8 = !scratch->w8.data.defined() || scratch->w8.data.numel() < w8Capacity;
        // Eager calls also replenish an exhausted reserve for a future first
        // capture on a framework-created side stream. No new reserve is needed
        // for a capture which already has its own W8 allocation.
        FusedControlReserve *controlReserve = nullptr;
        if (growW8 || !capturing) {
            controlReserve = &GetControlReserve(device, devOptI32, capturing);
        }

        // Capacity only affects mutable scratch. Accumulators begin at byte 0;
        // scales follow the aligned worst-case padded accumulator extent.
        const int64_t accRows = C + GMSQ_BM * experts;
        const int64_t accBytes = accRows * N * static_cast<int64_t>(sizeof(at::Half));
        const int64_t scBytes = experts * 2 * N2 * static_cast<int64_t>(sizeof(float));
        const int64_t accOffWs = 0;
        const int64_t scOffWs = (accBytes + 63) / 64 * 64;
        const int64_t wsTotal = scOffWs + scBytes;
        // at::empty uses the framework allocator during graph capture. No
        // pointer-table copy, format conversion or device synchronization occurs
        // here. Only graph-visible old allocations require permanent retention;
        // ordinary eager allocations keep the allocator's normal stream lifetime.
        if (!scratch->wsFlat.defined() || scratch->wsFlat.numel() < wsTotal) {
            at::Tensor replacement = at::empty({wsTotal}, devOptI8);
            if (scratch->wsCaptured) {
                scratch->retired.push_back(scratch->wsFlat);
            }
            scratch->wsFlat = replacement;
            scratch->wsCaptured = false;
        }
        if (growW8) {
            // Prewarm ALL layers before the first capture: reserve enough W8
            // for every static configuration already known on this device.
            FusedW8Allocation replacement;
            replacement.data = at::empty({w8Capacity}, devOptI8);
            TORCH_CHECK(reinterpret_cast<uintptr_t>(replacement.data.data_ptr()) % 32 == 0,
                        "grouped_matmul_situ_quant: W8 storage must be 32-byte aligned");
            const int64_t controlOffset = controlReserve->next * GMSQ_CONTROL_PAIR_WORDS;
            replacement.owner = controlReserve->freeBlock.narrow(
                0, controlOffset, GMSQ_CONTROL_LINE_WORDS);
            replacement.decision = controlReserve->freeBlock.narrow(
                0, controlOffset + GMSQ_CONTROL_LINE_WORDS, GMSQ_CONTROL_LINE_WORDS);
            ++controlReserve->next;
            scratch->w8 = std::move(replacement);
            scratch->w8Captured = false;
        }
        scratch->wsCaptured = scratch->wsCaptured || capturing;
        scratch->w8Captured = scratch->w8Captured || capturing;

        at::Tensor wPtrTf = fc.tbl.narrow(0, 0, static_cast<int64_t>(experts * 2));
        at::Tensor scPtrTf = fc.tbl.narrow(0, fc.scOff, static_cast<int64_t>(experts * 2));
        at::Tensor accWs = scratch->wsFlat.narrow(0, accOffWs, accBytes).view(at::kHalf);
        at::Tensor scaleF32 = scratch->wsFlat.narrow(0, scOffWs, scBytes).view(at::kFloat);
        at::Tensor wInt8 = scratch->w8.data;
        at::Tensor w8Owner = scratch->w8.owner;
        at::Tensor w8Decision = scratch->w8.decision;
        const uint64_t configId = fc.configId;
        // Tensor copies above are stable even if another host thread grows the
        // cache after unlocking; graph-visible allocations are never released.
        cacheLock.unlock();

        int32_t E32f = static_cast<int32_t>(experts);
        int32_t NP32f = static_cast<int32_t>(NP);
        int32_t N32f = static_cast<int32_t>(N);
        int32_t N232f = static_cast<int32_t>(N2);
        int32_t K32f = static_cast<int32_t>(K);
        int32_t C32f = static_cast<int32_t>(C);
        int32_t glType32 = static_cast<int32_t>(group_list_type);
        int32_t nzInput32 = srcIsNz ? 1 : 0;

        // P54: N-block scheduling was computed once per cache build (see the
        // cache-miss path above); the steady-state launch path just reads it.
        int32_t gemmNBlock = fc.gemmNBlock;

        int32_t kNumV = fc.kNum;
        int32_t nNum256V = fc.nNum256;
        uint32_t fusedBlockDim = static_cast<uint32_t>(aicCoreNum);
        // Snapshot launch addresses before returning to Python: a later set_()
        // can replace an input TensorImpl's storage before the task queue runs.
        // Storage refs keep the original buffers alive without retaining the
        // Python-owned TensorImpls in a callback destroyed on another thread.
        // StorageImpl can itself have a Python wrapper; do not assume this is
        // unconditionally GIL-free or create such wrappers on this path.
        void *xPtr = x.data_ptr();
        void *wPtrTfPtr = wPtrTf.data_ptr();
        void *scPtrTfPtr = scPtrTf.data_ptr();
        void *wInt8Ptr = wInt8.data_ptr();
        void *accWsPtr = accWs.data_ptr();
        void *scaleF32Ptr = scaleF32.data_ptr();
        void *xScalePtr = x_scale.data_ptr();
        void *yPtr = y.data_ptr();
        void *yScalePtr = y_scale.data_ptr();
        void *groupListPtr = kernelGroupList.data_ptr();
        void *w8OwnerPtr = w8Owner.data_ptr();
        void *w8DecisionPtr = w8Decision.data_ptr();
        std::vector<c10::Storage> keepAlive;
        keepAlive.reserve(12);
        keepAlive.emplace_back(x.storage());
        keepAlive.emplace_back(wPtrTf.storage());
        keepAlive.emplace_back(scPtrTf.storage());
        keepAlive.emplace_back(wInt8.storage());
        keepAlive.emplace_back(accWs.storage());
        keepAlive.emplace_back(scaleF32.storage());
        keepAlive.emplace_back(x_scale.storage());
        keepAlive.emplace_back(y.storage());
        keepAlive.emplace_back(y_scale.storage());
        keepAlive.emplace_back(kernelGroupList.storage());
        keepAlive.emplace_back(w8Owner.storage());
        keepAlive.emplace_back(w8Decision.storage());

        std::function<int()> gmsqHandler =
            [gmsqStream, fusedBlockDim, xPtr, wPtrTfPtr, scPtrTfPtr, wInt8Ptr, accWsPtr,
             scaleF32Ptr, xScalePtr, yPtr, yScalePtr, groupListPtr, w8OwnerPtr, w8DecisionPtr,
             keepAlive = std::move(keepAlive), E32f, kNumV, nNum256V, gemmNBlock, NP32f,
             N32f, N232f, K32f, C32f, glType32, nzInput32, betaF, invBeta, hasLinear, lbF, invLb,
             configId]() -> int {
                // Lifetime-only capture, released through the OpApi queue path.
                (void)keepAlive;
                gmsq_fused_256_cached_impl(fusedBlockDim, gmsqStream, xPtr, wPtrTfPtr, scPtrTfPtr,
                                     wInt8Ptr, accWsPtr, scaleF32Ptr, xScalePtr, yPtr,
                                     yScalePtr, groupListPtr, E32f, kNumV, nNum256V,
                                     gemmNBlock, NP32f, N32f, N232f, K32f, C32f,
                                     glType32, betaF, invBeta, hasLinear, lbF, invLb,
                                     nzInput32, w8OwnerPtr, w8DecisionPtr, configId);
                return 0;
            };
        // Use EXECUTE_OPAPI's handler release path. Do not construct an unused
        // OpCommand instance: its internal stack push requires a matching Run().
        at_npu::native::OpCommand::RunOpApi("grouped_matmul_situ_quant", gmsqHandler,
                                          /*sync=*/false);
        return {y, y_scale};
    }

    // The fused single-launch path is the only path: TORCH_CHECK(N2 %% GMSQ_BN == 0)
    // above (GMSQ_BN=128) forces N %% 256 == 0, and fusedEligible gates nNum256 <= 128.
    // No 4-kernel fallback exists in this MMImplType delivery.
    TORCH_CHECK(fusedEligible,
                "unsupported shape for fused kernel: N must be a multiple of 256 and N/256 <= 128");
    return {y, y_scale};
}

}  // namespace vllm_ascend
