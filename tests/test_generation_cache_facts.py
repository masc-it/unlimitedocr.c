from __future__ import annotations

import json

from unlimitedocr_c.frontend import prepare_text, project_root


def _context_text(name: str) -> str:
    return (project_root() / "data/context" / name).read_text(encoding="utf-8")


def test_frontend_converts_upstream_total_max_length_to_max_new_tokens() -> None:
    baseline = prepare_text("hello", max_new_tokens=0)
    total_budget = baseline.n_tokens + 3

    request = prepare_text("hello", max_length=total_budget, max_new_tokens=None)
    assert request.max_length == total_budget
    assert request.n_tokens == baseline.n_tokens
    assert request.max_new_tokens == 3

    exhausted = prepare_text("hello", max_length=baseline.n_tokens - 1, max_new_tokens=None)
    assert exhausted.max_new_tokens == 0

    explicit = prepare_text("hello", max_length=baseline.n_tokens, max_new_tokens=5)
    assert explicit.max_new_tokens == 5


def test_cached_upstream_generation_disables_hf_sliding_window_cache_truncation() -> None:
    config = json.loads(_context_text("config.json"))
    assert config["sliding_window"] == 128
    assert config["sliding_window_size"] == 128

    source = _context_text("modeling_unlimitedocr.py")
    assert "max_length=max_length" in source
    assert "self.config._ring_window = _orig_sw" in source
    assert "self.config.sliding_window = None" in source
    assert "self.config.sliding_window = _orig_sw" in source
    assert "Disable config.sliding_window to prevent DynamicCache from truncating prefill tokens" in source


def test_cached_upstream_sliding_attention_keeps_prefill_and_rings_only_generated_tokens() -> None:
    source = _context_text("modeling_deepseekv2.py")

    assert "W = getattr(self.config, '_ring_window', None)" in source
    assert "_is_true_prefill = (W is None or past_kv is None or" in source
    assert "q_len > 1 and (not hasattr(past_kv, '_prefill_length')" in source
    assert "past_kv._prefill_length[self.layer_idx] = _get_kcache(past_kv, self.layer_idx).shape[-2]" in source
    assert "if cur_len < prefill_len + W:" in source
    assert "slot = prefill_len + ring_pos" in source
    assert "k = _llama_repeat_kv(kcache, num_kv_groups)" in source
