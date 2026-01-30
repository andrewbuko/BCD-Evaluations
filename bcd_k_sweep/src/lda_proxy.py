#!/usr/bin/env python3
"""
Logit Difference Amplification (LDA) sampler to run the last step without whole thing due to hardware constraints

Implements LDA using the formula of:
    logits_amp = logits_after + alpha * (logits_after - logits_before)

Design goals:
- Correct KV-cache usage (no attention_mask shape gotchas)
- Apple Silicon (MPS) friendly + CUDA + CPU
- Robust dtype selection with fallback

Notes:
- "before" and "after" must be tokenizer/vocab-compatible. Best: same base family + same tokenizer.
- This file is meant to be imported by your experiment runner:
    from lda_proxy import generate_with_lda_proxy
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("lda_proxy")

# -----------------------
# Cache
# -----------------------
@dataclass
class CachedModelPair:
    model_before: PreTrainedModel
    model_after: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: torch.device
    dtype: torch.dtype
    vocab_size: int
    pad_token_id: int
    eos_token_id: Optional[int]


_MODEL_PAIR_CACHE: Dict[Tuple[str, str, str, str], CachedModelPair] = {}
# key = (before_id, after_id, device.type, dtype_str)


# -----------------------
# Device / dtype selection
# -----------------------
def get_best_device(prefer_mps: bool = True) -> torch.device:
    # Priority: MPS > CUDA > CPU (unless prefer_mps=False)
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def choose_dtype(device: torch.device, prefer_bf16_on_mps: bool = True) -> torch.dtype:
    """
    - MPS: prefer bf16 if supported; else fp16
    - CUDA: fp16
    - CPU: fp32
    """
    if device.type == "mps":
        if prefer_bf16_on_mps:
            # bfloat16 exists on many torch builds, but some ops/models can still choke.
            # We'll attempt bf16 at load time and fallback if needed.
            return torch.bfloat16
        return torch.float16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


# -----------------------
# Loading with fallback
# -----------------------
def _load_tokenizer(model_id: str) -> PreTrainedTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        # Most causal LMs can safely pad with EOS
        tok.pad_token = tok.eos_token
    # We don't actually pad in generation, but set anyway for safety
    tok.padding_side = "left"
    return tok


def _load_model(model_id: str, device: torch.device, dtype: torch.dtype) -> PreTrainedModel:
    # low_cpu_mem_usage reduces peak RAM during load
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def load_model_pair(
    model_before_id: str,
    model_after_id: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    prefer_mps: bool = True,
    prefer_bf16_on_mps: bool = True,
) -> CachedModelPair:
    dev = device or get_best_device(prefer_mps=prefer_mps)
    dt = dtype or choose_dtype(dev, prefer_bf16_on_mps=prefer_bf16_on_mps)
    key = (model_before_id, model_after_id, dev.type, _dtype_str(dt))

    if key in _MODEL_PAIR_CACHE:
        return _MODEL_PAIR_CACHE[key]

    logger.info(f"Loading LDA model pair on {dev.type} dtype={_dtype_str(dt)}")
    logger.info(f"  before: {model_before_id}")
    logger.info(f"  after : {model_after_id}")

    tok = _load_tokenizer(model_after_id)

    # Load with dtype fallback (esp. MPS bf16)
    tried: list[torch.dtype] = []
    last_err: Optional[Exception] = None

    def attempt(load_dtype: torch.dtype) -> Optional[CachedModelPair]:
        nonlocal last_err
        tried.append(load_dtype)
        try:
            mb = _load_model(model_before_id, dev, load_dtype)
            ma = _load_model(model_after_id, dev, load_dtype)

            # Check vocab compatibility
            vocab_b = mb.get_input_embeddings().weight.shape[0]
            vocab_a = ma.get_input_embeddings().weight.shape[0]
            if vocab_b != vocab_a:
                raise ValueError(
                    f"Vocab size mismatch: before={vocab_b}, after={vocab_a}. "
                    f"Models must share tokenizer/vocab for LDA."
                )

            # Extra sanity: tokenizer vocab should match model vocab
            if hasattr(tok, "vocab_size") and tok.vocab_size != vocab_a:
                logger.warning(
                    f"Tokenizer vocab_size ({tok.vocab_size}) != model vocab_size ({vocab_a}). "
                    "This can still work in some configs, but it often indicates mismatch."
                )

            cached = CachedModelPair(
                model_before=mb,
                model_after=ma,
                tokenizer=tok,
                device=dev,
                dtype=load_dtype,
                vocab_size=vocab_a,
                pad_token_id=int(tok.pad_token_id),
                eos_token_id=int(tok.eos_token_id) if tok.eos_token_id is not None else None,
            )
            return cached
        except Exception as e:
            last_err = e
            # If we partially loaded a model, try to free memory
            try:
                if "mb" in locals():
                    mb.cpu()
                    del mb
                if "ma" in locals():
                    ma.cpu()
                    del ma
            except Exception:
                pass
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            elif dev.type == "mps":
                torch.mps.empty_cache()
            gc.collect()
            return None

    # Primary attempt
    cached = attempt(dt)

    # Fallbacks
    if cached is None:
        if dev.type == "mps" and dt == torch.bfloat16:
            logger.warning(f"bf16 load failed on MPS ({last_err}); falling back to float16")
            cached = attempt(torch.float16)

        if cached is None and dev.type != "cpu":
            logger.warning(f"device/dtype load failed ({last_err}); falling back to CPU float32")
            cached_cpu = attempt(torch.float32)
            if cached_cpu is not None:
                cached = cached_cpu

    if cached is None:
        tried_str = ", ".join(_dtype_str(x) for x in tried)
        raise RuntimeError(f"Failed to load model pair after trying dtypes [{tried_str}]. Last error: {last_err}")

    _MODEL_PAIR_CACHE[key] = cached
    logger.info(f"Loaded successfully (vocab_size={cached.vocab_size})")
    return cached


# -----------------------
# Sampling helpers
# -----------------------
def _top_p_sample(probs_1d: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    probs_1d: shape (V,), sums to 1
    returns: shape (1,), token id
    """
    if top_p >= 1.0:
        return torch.multinomial(probs_1d, num_samples=1)

    sorted_probs, sorted_idx = torch.sort(probs_1d, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)

    remove = cum > top_p
    # keep at least 1 token
    remove[1:] = remove[:-1].clone()
    remove[0] = False

    filtered = sorted_probs.masked_fill(remove, 0.0)
    denom = filtered.sum()

    if denom.item() <= 1e-12:
        # Fallback: uniform
        filtered = torch.ones_like(filtered) / filtered.numel()
    else:
        filtered = filtered / denom

    sampled_pos = torch.multinomial(filtered, num_samples=1)
    token_id = sorted_idx.gather(-1, sampled_pos)
    return token_id


# -----------------------
# Core: LDA generation
# -----------------------
@torch.no_grad()
def generate_with_lda_proxy(
    prompt: str,
    alpha: float,
    model_before: str,
    model_after: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_cache: bool = True,
    stop_on_eos: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    retries: int = 1,  # kept for compatibility with your runner signature
) -> Dict[str, Any]:
    """
    Returns:
        {
          ok: bool,
          generated_text: str,
          n_tokens: int,
          avg_logprob_amp: float,
          error: str (if ok=False)
        }
    """
    if alpha < 0:
        return {"ok": False, "error": "alpha must be >= 0"}
    if temperature <= 0:
        return {"ok": False, "error": "temperature must be > 0"}
    if not (0.0 < top_p <= 1.0):
        return {"ok": False, "error": "top_p must be in (0, 1]"}
    if max_new_tokens <= 0:
        return {"ok": False, "error": "max_new_tokens must be > 0"}

    try:
        cached = load_model_pair(model_before, model_after, device=device, dtype=dtype)
        mb, ma, tok, dev = cached.model_before, cached.model_after, cached.tokenizer, cached.device

        # Tokenize (NO padding; simplest + safest for cache)
        inputs = tok(prompt, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(dev)

        generated = input_ids
        past_a = None
        past_b = None

        logprobs: list[float] = []

        for step in range(max_new_tokens):
            # Correct KV-cache usage:
            # - first step: pass full prompt, get cache
            # - subsequent: pass only last token, pass past_key_values, DO NOT pass attention_mask
            if use_cache and past_a is not None:
                model_input = generated[:, -1:]
                out_a = ma(model_input, past_key_values=past_a, use_cache=True)
                out_b = mb(model_input, past_key_values=past_b, use_cache=True)
            else:
                model_input = generated
                out_a = ma(model_input, use_cache=use_cache)
                out_b = mb(model_input, use_cache=use_cache)

            if use_cache:
                past_a = out_a.past_key_values
                past_b = out_b.past_key_values

            logits_a = out_a.logits[:, -1, :]  # (1, V)
            logits_b = out_b.logits[:, -1, :]  # (1, V)

            # LDA
            logits_amp = logits_a + alpha * (logits_a - logits_b)

            # temperature
            logits_amp = logits_amp / max(temperature, 1e-6)

            probs = F.softmax(logits_amp, dim=-1).squeeze(0)  # (V,)
            next_token = _top_p_sample(probs, top_p=top_p)  # (1,)
            next_id = int(next_token.item())

            # logprob under amplified distribution
            logprobs.append(float(torch.log(probs[next_id] + 1e-12).item()))

            # append
            generated = torch.cat([generated, next_token.view(1, 1).to(dev)], dim=-1)

            # stopping
            if stop_on_eos and cached.eos_token_id is not None and next_id == cached.eos_token_id:
                break

            # progress callback (decode only new tokens)
            if progress_callback is not None:
                new_tokens = generated[0, input_ids.shape[1] :]
                progress_callback(step + 1, tok.decode(new_tokens, skip_special_tokens=True))

        new_tokens = generated[0, input_ids.shape[1] :]
        text = tok.decode(new_tokens, skip_special_tokens=True)

        avg_logprob = (sum(logprobs) / len(logprobs)) if logprobs else 0.0
        return {
            "ok": True,
            "generated_text": text,
            "n_tokens": int(new_tokens.shape[0]),
            "avg_logprob_amp": float(avg_logprob),
        }

    except Exception as e:
        logger.exception("LDA generation failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# -----------------------
# Cache utilities
# -----------------------
def clear_cache() -> None:
    """Clear model cache and attempt to free device memory."""
    global _MODEL_PAIR_CACHE
    for pair in list(_MODEL_PAIR_CACHE.values()):
        try:
            pair.model_before.cpu()
            pair.model_after.cpu()
        except Exception:
            pass
    _MODEL_PAIR_CACHE.clear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    gc.collect()
    logger.info("Cleared model cache")


def get_cache_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "num_pairs": len(_MODEL_PAIR_CACHE),
        "pairs": [{"before": k[0], "after": k[1], "device": k[2], "dtype": k[3]} for k in _MODEL_PAIR_CACHE.keys()],
    }
    if torch.cuda.is_available():
        info["cuda_allocated_gb"] = float(torch.cuda.memory_allocated() / 1024**3)
        info["cuda_reserved_gb"] = float(torch.cuda.memory_reserved() / 1024**3)
    if torch.backends.mps.is_available():
        # current_allocated_memory exists on newer torch; if absent, skip
        try:
            info["mps_allocated_gb"] = float(torch.mps.current_allocated_memory() / 1024**3)
        except Exception:
            pass
    return info


# -----------------------
# Convenience wrapper
# -----------------------
def lda_generate(prompt: str, before: str, after: str, alpha: float = 0.5, **kwargs: Any) -> str:
    res = generate_with_lda_proxy(prompt=prompt, alpha=alpha, model_before=before, model_after=after, **kwargs)
    if not res.get("ok"):
        raise RuntimeError(res.get("error", "unknown error"))
    return str(res["generated_text"])


# -----------------------
# Minimal self-test
# -----------------------
if __name__ == "__main__":
    # IMPORTANT: Use compatible "before/after" pairs in real usage.
    # This self-test uses the SAME model for both, just to verify the loop runs.
    before_id = "gpt2"
    after_id = "gpt2"

    out = generate_with_lda_proxy(
        prompt="The key to effective leadership is",
        alpha=1.0,
        model_before=before_id,
        model_after=after_id,
        max_new_tokens=60,
        temperature=0.9,
        top_p=0.95,
    )

    if out["ok"]:
        print("=== GENERATED ===")
        print(out["generated_text"])
        print("\nmeta:", {k: out[k] for k in ("n_tokens", "avg_logprob_amp")})
        print("\ncache:", get_cache_info())
    else:
        print("ERROR:", out["error"])
