// Unlimited-OCR-2 DASH, fixed checkpoint dimensions. No dense score matrix.
#include "common.metal"

struct UocrDashParams {
    uint n_keys;
    uint query_start;
    uint n_queries;
    uint first_block;
};
struct UocrDashSelection {
    int blocks[12];
    float bias[12];
};

// Gemma RMSNorm must add one in FP32 before scaling, not pre-round 1+w
// into an FP16 weight. A single SIMD group covers each 128-dimensional row.
[[max_total_threads_per_threadgroup(32)]]
kernel void uocr_dash_norm_f16_inplace(device half *q [[buffer(0)]],
                                      device half *k [[buffer(1)]],
                                      device const half *q_weight [[buffer(2)]],
                                      device const half *k_weight [[buffer(3)]],
                                      uint2 group [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    device half *src = group.y == 0u ? q : k;
    device const half *weight = group.y == 0u ? q_weight : k_weight;
    float x[4], sum = 0;
    for (uint j = 0; j < 4; ++j) {
        x[j] = float(src[group.x*128u + lane + j*32u]);
        sum += x[j]*x[j];
    }
    const float scale = rsqrt(simd_sum(sum)/128.0f + 1e-6f);
    for (uint j = 0; j < 4; ++j)
        src[group.x*128u + lane + j*32u] = half(x[j]*scale*(1.0f + float(weight[lane+j*32u])));
}

// Router activations are nan_to_num'd AFTER norm/RoPE upstream. Inspect bits
// so FAST math cannot remove the finite check. Main attention is unchanged.
static inline float uocr_dash_value(half x) {
    return (as_type<ushort>(x) & 0x7c00u) == 0x7c00u ? 0.0f : float(x);
}

// One SIMD group per changed 32-token block; refresh only the mutable tail
// during decode. Completed block summaries and page ranges remain resident.
[[max_total_threads_per_threadgroup(32)]]
kernel void uocr_dash_summary_f16(device const half *keys [[buffer(0)]],
                                  device const half *head_cls [[buffer(1)]],
                                  device const UocrPagePosition *positions [[buffer(2)]],
                                  device float *summary [[buffer(3)]],
                                  device int2 *page_range [[buffer(4)]],
                                  constant UocrDashParams &p [[buffer(5)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    const uint block = p.first_block + group;
    const uint end = min((block + 1u) * 32u, p.n_keys);
    float hc[4], acc[4];
    for (uint j = 0; j < 4; ++j) { hc[j] = float(head_cls[lane + j*32]); acc[j] = 0; }
    float m = -INFINITY, denom = 0;
    int lo = 2147483647, hi = -1;
    for (uint t = block * 32u; t < end; ++t) {
        float kv[4], dot = 0;
        for (uint j = 0; j < 4; ++j) {
            kv[j] = uocr_dash_value(keys[t*128u + lane + j*32u]);
            dot += kv[j] * hc[j];
        }
        const float score = clamp(simd_sum(dot) * 0.08838834764831845f, -1e4f, 1e4f);
        const float next_m = max(m, score), corr = exp(m-next_m), e = exp(score-next_m);
        for (uint j = 0; j < 4; ++j) acc[j] = acc[j]*corr + kv[j]*e;
        denom = denom*corr + e;
        m = next_m;
        if (lane == 0u) {
            const int page = positions[t].image_page;
            if (page >= 0) { lo = min(lo, page); hi = max(hi, page); }
        }
    }
    for (uint j = 0; j < 4; ++j) summary[block*128u + lane + j*32u] = acc[j] / max(denom, 1e-6f);
    if (lane == 0u) page_range[block] = int2(hi >= 0 ? lo : -1, hi);
}

[[max_total_threads_per_threadgroup(32)]]
kernel void uocr_dash_route_f16(device const half *queries [[buffer(0)]],
                                device const float *summary [[buffer(1)]],
                                device const int2 *page_range [[buffer(2)]],
                                device const UocrPagePosition *positions [[buffer(3)]],
                                device UocrDashSelection *selection [[buffer(4)]],
                                constant UocrDashParams &p [[buffer(5)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off)
    const uint nb = (p.n_keys + 31u) / 32u;
    const int remote_last = int((p.query_start + row) / 32u) - 4;
    const int qp = positions[p.query_start + row].query_page;
    bool has_same = false;
    for (int b = int(lane); b <= remote_last; b += 32) {
        const int2 range = page_range[b];
        has_same |= qp >= 0 && range.x >= 0 && range.x <= qp && qp <= range.y;
    }
    has_same = simd_any(has_same);
    float q[4], top[16];
    int ids[16];
    for (uint j = 0; j < 4; ++j) q[j] = uocr_dash_value(queries[row*128u + lane + j*32u]);
    if (lane == 0u) for (uint j = 0; j < 16; ++j) { top[j] = -INFINITY; ids[j] = -1; }
    // Keep masked finite sentinels in top-M too, as the HF inference oracle
    // does. They can outrank extremely negative valid candidates. Stable ties
    // prefer the earlier block; upstream topk does not specify tie order.
    for (uint b = 0; b < nb; ++b) {
        const int2 range = page_range[b];
        const bool same = qp >= 0 && range.x >= 0 && range.x <= qp && qp <= range.y;
        const bool valid = int(b) <= remote_last && (!has_same || same);
        float x = -1e4f;
        if (valid) {
            float dot = 0;
            for (uint j = 0; j < 4; ++j) dot += q[j] * summary[b*128u + lane + j*32u];
            const float score = clamp(simd_sum(dot) * 0.08838834764831845f, -1e4f, 1e4f);
            x = score * 11.313708498984761f; // checkpoint scale 128 / sqrt(128)
        }
        if (lane == 0u) {
            for (uint j = 0; j < 16; ++j) {
                if (x > top[j]) {
                    for (uint t = 15; t > j; --t) { top[t] = top[t-1]; ids[t] = ids[t-1]; }
                    top[j] = x; ids[j] = valid ? int(b) : -1;
                    break;
                }
            }
        }
    }
    if (lane != 0u) return;
    const uint width = min(nb, 16u), kept = min(width, 12u);
    float z[16], tau[16], sum = 0, sumsq = 0;
    uint support = 0;
    for (uint j = 0; j < width; ++j) {
        z[j] = (top[j] - top[0]) * 0.5f;
        sum += z[j]; sumsq += z[j]*z[j];
        const float rho = float(j+1u), mean = sum/rho, mean_sq = sumsq/rho;
        const float ss = rho * (mean_sq - mean*mean);
        tau[j] = mean - sqrt(max((1.0f-ss)/rho, 0.0f));
        support += tau[j] <= z[j] ? 1u : 0u;
    }
    const float threshold = tau[max(support, 1u)-1u];
    float logp[12], mu = 0;
    uint count = 0;
    for (uint j = 0; j < 12; ++j) {
        const bool valid = j < kept && ids[j] >= 0;
        selection[row].blocks[j] = valid ? ids[j] : -1;
        const float mass = j < kept ? max(z[j] - threshold, 0.0f) : 0.0f;
        logp[j] = log(max(mass*mass, 1e-4f));
        if (valid) { mu += logp[j]; ++count; }
    }
    mu /= float(max(count, 1u));
    for (uint j = 0; j < 12; ++j)
        selection[row].bias[j] = selection[row].blocks[j] >= 0 ? logp[j] - mu : 0.0f;
}

// One SIMD group per query/head. Sparse online softmax reads at most 512
// keys, with no materialized attention matrix or gathered K/V scratch.
[[max_total_threads_per_threadgroup(32)]]
kernel void uocr_dash_attend_f16(device const half *queries [[buffer(0)]],
                                 device const half *keys [[buffer(1)]],
                                 device const half *values [[buffer(2)]],
                                 device const UocrDashSelection *selection [[buffer(3)]],
                                 device half *output [[buffer(4)]],
                                 constant UocrDashParams &p [[buffer(5)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x, head = group.y, pos = p.query_start + row;
    const uint local_start = pos >= 127u ? pos - 127u : 0u;
    float q[4], acc[4];
    for (uint j = 0; j < 4; ++j) { q[j] = float(queries[(row*10u+head)*128u+lane+j*32u]); acc[j] = 0; }
    float m = -INFINITY, denom = 0;
    for (uint section = 0; section < 13u; ++section) {
        const bool local = section == 12u;
        const int block = local ? 0 : selection[row].blocks[section];
        if (block < 0) continue;
        const uint begin = local ? local_start : uint(block)*32u;
        // Local tokens win on overlap and are visited exactly once, without bias.
        const uint end = local ? pos+1u : min((uint(block)+1u)*32u, local_start);
        const float bias = local ? 0.0f : selection[row].bias[section];
        for (uint t = begin; t < end; ++t) {
            float dot = 0;
            for (uint j = 0; j < 4; ++j) dot += q[j]*float(keys[(t*10u+head)*128u+lane+j*32u]);
            const float score = simd_sum(dot)*0.08838834764831845f + bias;
            const float next_m = max(m, score), corr = exp(m-next_m), e = exp(score-next_m);
            for (uint j = 0; j < 4; ++j)
                acc[j] = acc[j]*corr + e*float(values[(t*10u+head)*128u+lane+j*32u]);
            denom = denom*corr + e; m = next_m;
        }
    }
    for (uint j = 0; j < 4; ++j) output[(row*10u+head)*128u+lane+j*32u] = half(acc[j]/denom);
}
