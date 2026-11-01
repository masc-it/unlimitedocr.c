/* Document-window orchestration. Included by uocr_metal.m to share its private
 * encoder/cache helpers. Retained K/V is copied through shared memory only at
 * completed command-buffer boundaries; no weights or logits are duplicated.
 */
#define DOC_ROWS (1u + 159u + 274u + 128u)
#define DOC_MAIN_BYTES (UOCR_HIDDEN_SIZE * sizeof(uint16_t))
#define DOC_LAYER_BYTES ((uint64_t)DOC_ROWS * (2u * DOC_MAIN_BYTES + 256u))

typedef struct uocr_metal_document {
    uint64_t next_page;
    uint32_t max_page_tokens, owned, lookahead, pending, finished;
    uint32_t prefix_count, text_count;
    int32_t prefix[1u + 159u + 274u], text[128];
    uint16_t last_visual_row[UOCR_HIDDEN_SIZE];
    uint8_t kv[UOCR_DECODER_LAYERS * DOC_LAYER_BYTES];
} uocr_metal_document;

uint64_t uocr_metal_document_bytes(const uocr_metal_context *ctx) {
    return ctx != NULL && ctx->document != NULL ? sizeof(uocr_metal_document) : 0u;
}

uint64_t uocr_metal_vision_cached_position_bytes(const uocr_metal_context *ctx) {
    if (ctx == NULL) return 0u;
    return (uint64_t)[ctx->sam_abs_pos_local_buffer length] +
        (uint64_t)[ctx->sam_rel_pos_local_buffer length] +
        (uint64_t)[ctx->clip_abs_pos_local_buffer length];
}

uint64_t uocr_metal_document_vision_staging_bytes(const uocr_prepared_request *request) {
    uint64_t bytes = 0u;
    // Validated base views are uploaded together. The non-overlapping patch
    // extraction buffer holds the same pixels in FP16, alongside the upload.
    // Including the reused lookahead view is conservative on later windows.
    for (uint32_t i = 0u; i < request->n_views; ++i) {
        const uocr_image_view *v = &request->views[i];
        const uint64_t element_bytes = v->format == UOCR_PIXEL_F16_NCHW ? 2u : 4u;
        bytes += (uint64_t)3u * v->width * v->height * (element_bytes + sizeof(uint16_t));
    }
    return bytes;
}

int uocr_metal_document_begin(uocr_metal_context *ctx, uint32_t max_page_tokens,
    uint64_t budget, char *error, size_t error_size) {
    if (ctx == NULL || !ctx->model_is_v2 || ctx->document != NULL ||
        max_page_tokens == 0u || max_page_tokens > 20000u) {
        (void)metal_fail(error, error_size, "document session requires idle v2 Metal and a page limit in [1,20000]");
        return UOCR_ERROR_INVALID_ARGUMENT;
    }
    uocr_metal_context_release_v2_request(ctx);
    const uint64_t live = ctx->memory_tracker != NULL ? ctx->memory_tracker->total_live_bytes : 0u;
    uint64_t required;
    if (!checked_add_u64(live, uocr_metal_vision_cached_position_bytes(ctx), &required) ||
        !checked_add_u64(required, sizeof(uocr_metal_document), &required)) {
        (void)metal_fail(error, error_size, "document continuation memory requirement overflow");
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    if (budget != 0u && required > budget) {
        (void)metal_fail(error, error_size, "document continuation requires %llu bytes including model and cached positions; budget is %llu",
                        (unsigned long long)required, (unsigned long long)budget);
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    uocr_metal_document *d = calloc(1u, sizeof(*d));
    if (d == NULL) {
        (void)metal_fail(error, error_size, "failed to allocate document continuation state");
        return UOCR_ERROR_OUT_OF_MEMORY;
    }
    if (!metal_memory_reserve(ctx, UOCR_MEMORY_KV_CACHE, sizeof(*d), "document continuation state", error, error_size)) {
        free(d); return UOCR_ERROR_OUT_OF_MEMORY;
    }
    d->max_page_tokens = max_page_tokens;
    ctx->document = d;
    return UOCR_OK;
}

void uocr_metal_document_end(uocr_metal_context *ctx) {
    if (ctx == NULL || ctx->document == NULL) return;
    uocr_metal_context_release_v2_request(ctx);
    free(ctx->document); ctx->document = NULL;
    metal_memory_release(ctx, UOCR_MEMORY_KV_CACHE, sizeof(uocr_metal_document));
}

int uocr_metal_document_set_window(uocr_metal_context *ctx, uint64_t first_page,
    uint32_t owned_pages, uint32_t lookahead, char *error, size_t error_size) {
    uocr_metal_document *d = ctx != NULL ? ctx->document : NULL;
    if (d == NULL || d->pending || d->finished || first_page != d->next_page ||
        owned_pages == 0u || owned_pages > UOCR_V2_MAX_POSITIONS / 274u || lookahead > 1u ||
        first_page > UINT64_MAX - owned_pages || (first_page != 0u && d->prefix_count == 0u))
        return metal_fail(error, error_size, "document requires a complete contiguous predecessor and valid window");
    d->owned = owned_pages; d->lookahead = lookahead; d->pending = 1u;
    return 1;
}

int uocr_metal_document_validate_window(uocr_metal_context *ctx,
    const uocr_prepared_request *r, char *error, size_t error_size) {
    const uocr_metal_document *d = ctx->document;
    if (d == NULL || !d->pending || r->n_views != d->owned + d->lookahead || r->max_new_tokens == 0u ||
        (uint64_t)r->max_new_tokens > (uint64_t)d->owned * d->max_page_tokens + 256u)
        return metal_fail(error, error_size, "document window page count or output budget is invalid");
    for (uint32_t i = 0; i < r->n_views; ++i)
        if (r->views[i].kind != UOCR_VIEW_GLOBAL)
            return metal_fail(error, error_size, "document streaming supports base-mode pages only");
    // The rewrite relies on the standard contiguous PAGE/image sequence. Custom
    // instructions after the images remain allowed in the first window only.
    uint32_t pos = 1u;
    for (uint32_t page = 0; page < r->n_views; ++page) {
        if (pos >= r->n_tokens || r->input_ids[pos++] != UOCR_TOKEN_PAGE)
            return metal_fail(error, error_size, "document windows require standard PAGE/image prompts");
        for (uint32_t j = 0; j < UOCR_GLOBAL_VISUAL_TOKENS; ++j)
            if (pos >= r->n_tokens || r->input_ids[pos++] != UOCR_TOKEN_IMAGE)
                return metal_fail(error, error_size, "document window image span is incomplete");
    }
    return 1;
}

static uint8_t *metal_document_slot(uocr_metal_document *d, uint32_t layer, uint32_t kind, uint32_t row) {
    const uint32_t width = kind < 2u ? DOC_MAIN_BYTES : 256u;
    return d->kv + layer * DOC_LAYER_BYTES + (uint64_t)kind * DOC_ROWS * DOC_MAIN_BYTES + (uint64_t)row * width;
}

/* GPU work has completed before either direction of this copy. All retained
 * native cache buffers use shared storage. K/V are already RoPE-transformed.
 */
static void metal_document_copy(uocr_metal_context *ctx, uint32_t cached_row,
    uint32_t saved_row, uint32_t rows, int restore) {
    const uint32_t slots[3] = {DASH_K, DASH_V, DASH_ROUTER_K};
    for (uint32_t layer = 0u; layer < UOCR_DECODER_LAYERS; ++layer) {
        uocr_metal_buffer_slice slices[DASH_BUFFER_COUNT];
        metal_dash_cache_slices(ctx->dash_cache, layer, slices);
        for (uint32_t kind = 0u; kind < 3u; ++kind) {
            const uint32_t width = kind < 2u ? DOC_MAIN_BYTES : 256u;
            uint8_t *cached = (uint8_t *)[slices[slots[kind]].buffer contents] +
                slices[slots[kind]].offset + (uint64_t)cached_row * width;
            uint8_t *saved = metal_document_slot(ctx->document, layer, kind, saved_row);
            if (restore) memcpy(cached, saved, (size_t)rows * width);
            else memcpy(saved, cached, (size_t)rows * width);
        }
    }
}

static int metal_document_refresh(uocr_metal_context *ctx, char *error, size_t error_size) {
    @autoreleasepool {
        uocr_metal_dash_cache *c = ctx->dash_cache;
        const uint32_t slots[5] = {DASH_ROUTER_K, DASH_HEAD_CLS, DASH_POSITIONS, DASH_SUMMARIES, DASH_PAGE_RANGES};
        const uocr_metal_dash_params params = {c->length, 0u, c->length, 0u};
        if (!metal_command_batch_begin(ctx, error, error_size)) return 0;
        id<MTLComputePipelineState> pipeline = metal_get_pipeline(ctx, "uocr_dash_summary_cached_f16", error, error_size);
        if (pipeline == nil) goto failed;
        for (uint32_t layer = 0u; layer < UOCR_DECODER_LAYERS; ++layer) {
            uocr_metal_buffer_slice slices[DASH_BUFFER_COUNT];
            metal_dash_cache_slices(c, layer, slices);
            id<MTLComputeCommandEncoder> enc = metal_compute_command_encoder(ctx, ctx->active_command_buffer);
            if (enc == nil) { (void)metal_fail(error, error_size, "failed to encode restored DASH summaries"); goto failed; }
            [enc setComputePipelineState:pipeline];
            for (uint32_t i = 0; i < 5u; ++i)
                [enc setBuffer:slices[slots[i]].buffer offset:slices[slots[i]].offset atIndex:i];
            [enc setBytes:&params length:sizeof(params) atIndex:5u];
            [enc setBuffer:c->arena offset:(NSUInteger)(c->layout.tail_offset + layer * UOCR_METAL_DASH_TAIL_STRIDE) atIndex:6u];
            [enc dispatchThreadgroups:MTLSizeMake((c->length + 31u) / 32u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];
            metal_end_compute_encoder(ctx, enc);
        }
        return metal_command_batch_commit_and_wait(ctx, "document restored summaries", error, error_size);
    failed:
        metal_command_batch_abort(ctx); return 0;
    }
}

static int metal_document_retain(uocr_metal_context *ctx, const int32_t *prompt,
    uint32_t prompt_count, uint32_t context_pages, uint32_t previous_text_count,
    const uocr_metal_decoder_result_f16 *result, uocr_metal_buffer_slice last_visual,
    char *error, size_t error_size) {
    uocr_metal_document *d = ctx->document;
    uint32_t page = 0u, marker = UINT32_MAX, end = 0u;
    for (uint32_t i = 0; i < prompt_count; ++i) {
        if (prompt[i] == UOCR_TOKEN_PAGE && ++page == context_pages + d->owned) {
            marker = i; end = i + 1u;
            while (end < prompt_count && prompt[end] == UOCR_TOKEN_IMAGE) ++end;
            break;
        }
    }
    if (marker == UINT32_MAX || end < 128u + end % 32u ||
        ctx->dash_cache->length < prompt_count + previous_text_count + result->generated_count ||
        !metal_slice_valid(last_visual, DOC_MAIN_BYTES))
        return metal_fail(error, error_size, "document finished without a reusable computed tail");
    const uint32_t start = end - 128u - end % 32u;
    const uint32_t image_rows = end - start + 274u;
    if (start <= marker || end + 274u > prompt_count)
        return metal_fail(error, error_size, "document lookahead retention range is invalid");
    uint32_t last_marker = UINT32_MAX;
    for (uint32_t i = 0; i < result->generated_count; ++i)
        if (result->generated_ids[i] == UOCR_TOKEN_PAGE) last_marker = i;
    if (last_marker == UINT32_MAX)
        return metal_fail(error, error_size, "document output has no completed page");
    uint32_t text_start = last_marker + 1u;
    if (result->generated_count - text_start > 128u) text_start = result->generated_count - 128u;
    const uint32_t text_count = result->generated_count - text_start;
    metal_document_copy(ctx, marker, 0u, 1u, 0);
    metal_document_copy(ctx, start, 1u, image_rows, 0);
    metal_document_copy(ctx, prompt_count + previous_text_count + text_start, 1u + image_rows, text_count, 0);
    d->prefix_count = 1u + image_rows; d->text_count = text_count;
    d->prefix[0] = UOCR_TOKEN_PAGE;
    memcpy(d->prefix + 1u, prompt + start, (size_t)image_rows * sizeof(int32_t));
    memcpy(d->text, result->generated_ids + text_start, (size_t)text_count * sizeof(int32_t));
    // Vision output is GPU-private. The completed window no longer needs its
    // shared prompt arena, so use one row there to read back the final embedding.
    uocr_metal_buffer_slice readback; uint64_t bytes;
    if (!metal_prompt_arena_slice(ctx, 0u, 1u, &readback, &bytes))
        return metal_fail(error, error_size, "document embedding readback workspace is invalid");
    if (!metal_copy_slice(ctx, last_visual, readback, DOC_MAIN_BYTES,
            "document lookahead embedding", error, error_size)) return 0;
    memcpy(d->last_visual_row, (uint8_t *)[readback.buffer contents] + readback.offset, DOC_MAIN_BYTES);
    metal_profile_add_event_ms(ctx, "metal.document.tail_retain", 0.0);
    return 1;
}

static int metal_generate_document_window(uocr_metal_context *ctx,
    const uocr_prepared_request *r, uocr_metal_decoder_result_f16 *result,
    char *error, size_t error_size) {
    uocr_metal_document *d = ctx->document;
    if (!uocr_metal_document_validate_window(ctx, r, error, error_size)) return 0;
    const uint32_t reuse = d->prefix_count != 0u;
    const uint32_t saved_text = reuse ? d->text_count : 0u;
    const uint32_t fresh_pages = r->n_views - reuse;
    const uint32_t image_end = 1u + r->n_views * 274u;
    const uint32_t prompt_count = reuse ? d->prefix_count + (r->n_views - 1u) * 274u : r->n_tokens;
    const uint32_t copied = reuse ? d->prefix_count - (fresh_pages == 0u ? 1u : 0u) : 0u;
    const uint32_t query_count = prompt_count - copied;
    const uint32_t vision_tokens = 1u + fresh_pages * 274u;
    const uint64_t host_bytes = (uint64_t)(prompt_count + vision_tokens) * (sizeof(int32_t) + sizeof(uint8_t));
    if (prompt_count + saved_text > UOCR_V2_MAX_POSITIONS ||
        r->max_new_tokens > UOCR_V2_MAX_POSITIONS - prompt_count - saved_text ||
        result->generated_ids == NULL || result->generated_capacity < r->max_new_tokens)
        return metal_fail(error, error_size, "document window exceeds context/output capacity");
    uocr_metal_context_release_v2_request(ctx);
    if (!metal_memory_reserve(ctx, UOCR_MEMORY_TRANSIENT_BUFFERS, host_bytes, "document prompt metadata", error, error_size)) return 0;
    uint8_t *host = malloc((size_t)host_bytes);
    int ok = 0;
    if (host == NULL) { (void)metal_fail(error, error_size, "failed to allocate document prompt metadata"); goto cleanup; }
    int32_t *prompt = (int32_t *)host;
    int32_t *vision_ids = prompt + prompt_count;
    uint8_t *mask = (uint8_t *)(vision_ids + vision_tokens);
    uint8_t *vision_mask = mask + prompt_count;
    if (reuse) {
        memcpy(prompt, d->prefix, (size_t)d->prefix_count * sizeof(int32_t));
        memcpy(prompt + d->prefix_count, r->input_ids + 275u, (size_t)(image_end - 275u) * sizeof(int32_t));
    } else memcpy(prompt, r->input_ids, (size_t)prompt_count * sizeof(int32_t));
    for (uint32_t i = 0; i < prompt_count; ++i) mask[i] = prompt[i] == UOCR_TOKEN_IMAGE;
    vision_ids[0] = UOCR_TOKEN_BOS; vision_mask[0] = 0u;
    for (uint32_t i = 1; i < vision_tokens; ++i) {
        vision_mask[i] = (i - 1u) % 274u != 0u;
        vision_ids[i] = vision_mask[i] ? UOCR_TOKEN_IMAGE : UOCR_TOKEN_PAGE;
    }
    uocr_metal_buffer_slice visual = {nil, 0u}, last_visual = {nil, 0u};
    if (fresh_pages != 0u) {
        uocr_prepared_request vision_request = *r;
        vision_request.input_ids = vision_ids; vision_request.image_mask = vision_mask;
        vision_request.n_tokens = vision_tokens; vision_request.max_new_tokens = 0u;
        vision_request.views += reuse; vision_request.n_views = fresh_pages;
        uocr_vision_schedule schedule;
        if (uocr_plan_vision_schedule(&vision_request, fresh_pages, NULL, 0u, &schedule, error, error_size) != UOCR_OK) goto cleanup;
        uocr_metal_vision_workspace scratch;
        memset(&scratch, 0, sizeof(scratch));
        if (!metal_context_encode_visual_features_to_workspace_f16(ctx, &vision_request, fresh_pages,
                &schedule, &scratch, 0, error, error_size)) goto cleanup;
        visual = (uocr_metal_buffer_slice){scratch.final_visual_rows.buffer, scratch.final_visual_rows.offset};
        last_visual = visual;
        last_visual.offset += (NSUInteger)(fresh_pages * UOCR_GLOBAL_VISUAL_TOKENS - 1u) * DOC_MAIN_BYTES;
        uocr_metal_context_release_scratch(ctx, UOCR_METAL_SCRATCH_VISION);
        metal_vision_workspace_reset(&scratch);
    }
    if (reuse) metal_profile_add_event_ms(ctx, "metal.document.lookahead_reuse", 0.0);
    const uint32_t capacity = prompt_count + saved_text + r->max_new_tokens;
    const uint32_t pages = r->n_views + reuse + 1u;
    const uint64_t required = uocr_metal_v2_decoder_required_bytes(capacity, query_count, pages);
    if (!required || !uocr_metal_context_prepare_v2_decoder(ctx, capacity, query_count, pages,
            reuse ? d->next_page - 1u : d->next_page, required, error, error_size)) goto cleanup;
    uocr_metal_dash_cache *c = ctx->dash_cache;
    uocr_page_position *positions = (uocr_page_position *)((uint8_t *)[c->arena contents] + c->layout.offsets[DASH_POSITIONS]);
    if (reuse) {
        metal_document_copy(ctx, 0u, 0u, copied, 1);
        for (uint32_t i = 0; i < copied; ++i)
            if (uocr_page_state_step(&c->pages, prompt[i], 0, &positions[i], error, error_size) != UOCR_OK) goto cleanup;
        c->length = copied;
        if (!metal_document_refresh(ctx, error, error_size)) goto cleanup;
        metal_profile_add_event_ms(ctx, "metal.document.prefix_restore", 0.0);
    }
    uocr_metal_decoder_request_f16 pending;
    memset(&pending, 0, sizeof(pending));
    pending.input_ids = prompt + copied; pending.image_mask = mask + copied; pending.n_tokens = query_count;
    if (!uocr_metal_context_dash_begin(ctx, pending.input_ids, query_count, 0, error, error_size)) goto cleanup;
    ctx->reuse_compute_encoder = 1;
    if (fresh_pages == 0u) {
        uocr_metal_buffer_slice input; uint64_t bytes;
        if (!metal_prompt_arena_slice(ctx, 0u, 1u, &input, &bytes)) goto cleanup;
        memcpy((uint8_t *)[input.buffer contents] + input.offset, d->last_visual_row, DOC_MAIN_BYTES);
        if (!metal_command_batch_begin(ctx, error, error_size)) goto cleanup;
    } else if (!metal_command_batch_begin(ctx, error, error_size) ||
        !metal_assemble_v2_page_prompt(ctx, &pending, visual.buffer, visual.offset, error, error_size)) goto cleanup;
    if (!metal_encode_v2_arena(ctx, error, error_size) || !metal_commit_decoder_prefill(ctx, error, error_size)) goto cleanup;
    metal_decoder_result_reset(result);
    if (reuse) {
        metal_document_copy(ctx, c->length, d->prefix_count, saved_text, 1);
        for (uint32_t i = 0; i < saved_text; ++i)
            positions[c->length + i] = (uocr_page_position){0u, 1u + c->pages.page_base_parity, -1, 0};
        c->length += saved_text;
        // Match upstream continuation_carry: copied text does not advance the
        // replayed prompt counters; the forced marker skips the context page.
        ++c->pages.marker_count; c->pages.generated_started = 1u;
        if (!metal_document_refresh(ctx, error, error_size)) goto cleanup;
        metal_profile_add_event_ms(ctx, "metal.document.text_restore", 0.0);
        const int32_t marker = UOCR_TOKEN_PAGE;
        if (!uocr_metal_context_dash_begin(ctx, &marker, 1u, 1, error, error_size) ||
            !metal_command_batch_begin(ctx, error, error_size) ||
            !metal_context_assemble_prompt_from_model_to_arena_f16(ctx, &marker, 1u,
                UINT32_MAX, 0u, NULL, nil, 0u, 0u, error, error_size) ||
            !metal_encode_v2_arena(ctx, error, error_size)) goto cleanup;
        result->generated_ids[result->generated_count++] = marker;
    }
    const int marker_stop = reuse || d->lookahead;
    uocr_page_guard guard = {d->owned, d->max_page_tokens, reuse, 0u};
    int complete = 0;
    for (;;) {
        int32_t id; float score;
        uint32_t filter = uocr_page_guard_filter(&guard);
        // SGLang permits EOS on the final owned page. Only a non-reused final
        // window masks PAGE there; lookahead/reuse also accepts the extra marker
        // as a stop signal and removes it from both output and retained text.
        if (filter == 0u && !marker_stop && guard.markers == d->owned)
            filter = UOCR_TOKEN_PAGE + 1u;
        if (!metal_v2_argmax(ctx, filter, &id, &score, error, error_size)) goto cleanup;
        if (marker_stop && id == UOCR_TOKEN_PAGE && guard.markers == d->owned) { complete = 1; break; }
        if (id == UOCR_TOKEN_EOS) {
            if (guard.markers != d->owned) { (void)metal_fail(error, error_size, "document ended before its owned pages"); goto cleanup; }
            result->stopped_on_eos = 1u; complete = 1;
            break;
        }
        if (result->generated_count == r->max_new_tokens) break;
        result->generated_ids[result->generated_count++] = id;
        result->last_token_id = (uint32_t)id; result->last_score_f32 = score;
        uocr_page_guard_accept(&guard, id);
        // Feed the final allowed output too: EOS/the window stop marker does
        // not consume the emitted-token budget, but needs that last token's K/V.
        if (!uocr_metal_context_dash_begin(ctx, &id, 1u, 1, error, error_size) ||
            !metal_command_batch_begin(ctx, error, error_size) ||
            !metal_context_assemble_prompt_from_model_to_arena_f16(ctx, &id, 1u,
                UINT32_MAX, 0u, NULL, nil, 0u, 0u, error, error_size) ||
            !metal_encode_v2_arena(ctx, error, error_size)) goto cleanup;
    }
    if (!complete) { (void)metal_fail(error, error_size, "document window exhausted its output budget before completion"); goto cleanup; }
    if (d->lookahead && !metal_document_retain(ctx, prompt, prompt_count, reuse, saved_text,
            result, last_visual, error, error_size)) goto cleanup;
    d->next_page += d->owned; d->pending = 0u; d->finished = !d->lookahead;
    ok = 1;
cleanup:
    ctx->reuse_compute_encoder = 0;
    metal_command_batch_abort(ctx);
    free(host);
    metal_memory_release(ctx, UOCR_MEMORY_TRANSIENT_BUFFERS, host_bytes);
    uocr_metal_context_release_v2_request(ctx);
    return ok;
}
