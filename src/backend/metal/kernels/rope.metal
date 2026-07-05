// rope.metal - RoPE QK kernels
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrRopeQKParams {
    uint n_tokens;
    uint heads;
    uint head_dim;
    uint position_start;
    float freq_scale;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

static inline void uocr_rope_pair(uint gid,
                                  constant UocrRopeQKParams &params,
                                  uint heads,
                                  uint head_dim,
                                  float freq_scale,
                                  thread uint &token,
                                  thread uint &head,
                                  thread uint &a,
                                  thread uint &b,
                                  thread float &c,
                                  thread float &s) {
    const uint half_dim = head_dim >> 1u;
    const uint pair = gid % half_dim;
    const uint token_head = gid / half_dim;
    head = token_head % heads;
    token = token_head / heads;
    a = pair;
    b = pair + half_dim;
    const float angle = float(params.position_start + token) * exp2(float(pair) * freq_scale);
    c = cos(angle);
    s = sin(angle);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rope_qk_f16_to_f16(device const half *q_src [[buffer(0)]],
                                    device const half *k_src [[buffer(1)]],
                                    device half *q_dst [[buffer(2)]],
                                    device half *k_dst [[buffer(3)]],
                                    constant UocrRopeQKParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const float freq_scale = uocr_fc_rope_freq_scale_or(params.freq_scale);
    const uint half_dim = head_dim >> 1u;
    const uint total = params.n_tokens * heads * half_dim;
    if (gid >= total) {
        return;
    }

    uint token;
    uint head;
    uint a;
    uint b;
    float c;
    float s;
    uocr_rope_pair(gid, params, heads, head_dim, freq_scale, token, head, a, b, c, s);
    const ulong base = (ulong(token) * ulong(heads) + ulong(head)) * ulong(head_dim);
    const float q0 = float(q_src[base + ulong(a)]);
    const float q1 = float(q_src[base + ulong(b)]);
    const float k0 = float(k_src[base + ulong(a)]);
    const float k1 = float(k_src[base + ulong(b)]);
    q_dst[base + ulong(a)] = half(q0 * c - q1 * s);
    q_dst[base + ulong(b)] = half(q0 * s + q1 * c);
    k_dst[base + ulong(a)] = half(k0 * c - k1 * s);
    k_dst[base + ulong(b)] = half(k0 * s + k1 * c);
}

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_rope_qk_f16_to_f32(device const half *q_src [[buffer(0)]],
                                    device const half *k_src [[buffer(1)]],
                                    device float *q_dst [[buffer(2)]],
                                    device float *k_dst [[buffer(3)]],
                                    constant UocrRopeQKParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const float freq_scale = uocr_fc_rope_freq_scale_or(params.freq_scale);
    const uint half_dim = head_dim >> 1u;
    const uint total = params.n_tokens * heads * half_dim;
    if (gid >= total) {
        return;
    }

    uint token;
    uint head;
    uint a;
    uint b;
    float c;
    float s;
    uocr_rope_pair(gid, params, heads, head_dim, freq_scale, token, head, a, b, c, s);
    const ulong base = (ulong(token) * ulong(heads) + ulong(head)) * ulong(head_dim);
    const float q0 = float(q_src[base + ulong(a)]);
    const float q1 = float(q_src[base + ulong(b)]);
    const float k0 = float(k_src[base + ulong(a)]);
    const float k1 = float(k_src[base + ulong(b)]);
    q_dst[base + ulong(a)] = q0 * c - q1 * s;
    q_dst[base + ulong(b)] = q0 * s + q1 * c;
    k_dst[base + ulong(a)] = k0 * c - k1 * s;
    k_dst[base + ulong(b)] = k0 * s + k1 * c;
}

// Decode RoPE KV write
struct UocrDecodeRopeKvWriteParams {
    uint heads;
    uint head_dim;
    uint position;
    float freq_scale;
    uint batch_slots;
    uint cache_token_capacity;
    uint layer;
    uint slot;
    uint prompt_length;
    uint ring_window;
    uint reserved[2];
};

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_decode_rope_kv_write_one_f16(
                                    device half *q_dst [[buffer(0)]],
                                    device const half *k_src [[buffer(1)]],
                                    device const half *v_src [[buffer(2)]],
                                    device half *k_cache [[buffer(3)]],
                                    device half *v_cache [[buffer(4)]],
                                    constant UocrDecodeRopeKvWriteParams &params [[buffer(5)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const float freq_scale = uocr_fc_rope_freq_scale_or(params.freq_scale);
    const uint ring_window = uocr_fc_ring_window_or(params.ring_window);
    const uint half_dim = head_dim >> 1u;
    const uint total = heads * half_dim;
    if (gid >= total) {
        return;
    }
    const uint head = gid / half_dim;
    const uint pair = gid % half_dim;
    const uint a = pair;
    const uint b = pair + half_dim;
    const float angle = float(params.position) * exp2(float(pair) * freq_scale);
    const float c = cos(angle);
    const float s = sin(angle);

    /* RoPE Q in-place */
    const ulong base = (ulong(head)) * ulong(head_dim);
    const float q0 = float(q_dst[base + ulong(a)]);
    const float q1 = float(q_dst[base + ulong(b)]);
    q_dst[base + ulong(a)] = half(q0 * c - q1 * s);
    q_dst[base + ulong(b)] = half(q0 * s + q1 * c);

    /* RoPE K and write to cache */
    const float k0 = float(k_src[base + ulong(a)]);
    const float k1 = float(k_src[base + ulong(b)]);
    const uint position = params.position;
    uint cache_token = position;
    if (position >= params.prompt_length) {
        cache_token = params.prompt_length + ((position - params.prompt_length) % ring_window);
    }
    if (cache_token < params.cache_token_capacity) {
        const ulong dst_index = (((ulong(params.layer) * ulong(params.batch_slots) + ulong(params.slot)) *
                                  ulong(params.cache_token_capacity) + ulong(cache_token)) *
                                 ulong(heads) + ulong(head)) * ulong(head_dim);
        k_cache[dst_index + ulong(a)] = half(k0 * c - k1 * s);
        k_cache[dst_index + ulong(b)] = half(k0 * s + k1 * c);
        v_cache[dst_index + ulong(a)] = v_src[base + ulong(a)];
        v_cache[dst_index + ulong(b)] = v_src[base + ulong(b)];
    }
}
