// kv_cache.metal - KV cache write
// Extracted from uocr_smoke.metal
//
#include "common.metal"
struct UocrKVCacheWriteParams {
    uint n_tokens;
    uint batch_slots;
    uint cache_token_capacity;
    uint layer;
    uint slot;
    uint prompt_length;
    uint position_start;
    uint heads;
    uint head_dim;
    uint ring_window;
    uint reserved0;
    uint reserved1;
};

[[max_total_threads_per_threadgroup(256)]] kernel void uocr_kv_cache_write_f16(device const half *k_src [[buffer(0)]],
                                    device const half *v_src [[buffer(1)]],
                                    device half *k_cache [[buffer(2)]],
                                    device half *v_cache [[buffer(3)]],
                                    constant UocrKVCacheWriteParams &params [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
    const uint heads = uocr_fc_attention_heads_or(params.heads);
    const uint head_dim = uocr_fc_head_dim_or(params.head_dim);
    const uint ring_window = uocr_fc_ring_window_or(params.ring_window);
    const uint vec_head_dim = uocr_div_up_u32(head_dim, UOCR_HALF4_WIDTH);
    const uint head_area_vec = heads * vec_head_dim;
    const uint total_vec = params.n_tokens * head_area_vec;
    if (gid >= total_vec || vec_head_dim == 0u) {
        return;
    }

    const uint dim_vec = gid % vec_head_dim;
    const uint dim = dim_vec * UOCR_HALF4_WIDTH;
    const uint head = (gid / vec_head_dim) % heads;
    const uint token = gid / head_area_vec;
    const uint position = params.position_start + token;
    uint cache_token = position;
    if (position >= params.prompt_length) {
        cache_token = params.prompt_length + ((position - params.prompt_length) % ring_window);
    }
    if (cache_token >= params.cache_token_capacity) {
        return;
    }

    const ulong src_index = (ulong(token) * ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
    const ulong dst_index = (((ulong(params.layer) * ulong(params.batch_slots) + ulong(params.slot)) *
                              ulong(params.cache_token_capacity) + ulong(cache_token)) *
                             ulong(heads) + ulong(head)) * ulong(head_dim) + ulong(dim);
    if (dim + (UOCR_HALF4_WIDTH - 1u) < head_dim) {
        uocr_store_half4(k_cache, dst_index, uocr_load_half4(k_src, src_index));
        uocr_store_half4(v_cache, dst_index, uocr_load_half4(v_src, src_index));
    } else {
        for (uint i = 0u; i < UOCR_HALF4_WIDTH && dim + i < head_dim; ++i) {
            k_cache[dst_index + ulong(i)] = k_src[src_index + ulong(i)];
            v_cache[dst_index + ulong(i)] = v_src[src_index + ulong(i)];
        }
    }
}
