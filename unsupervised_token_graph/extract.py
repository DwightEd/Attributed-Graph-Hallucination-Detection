"""Extract label-free traces from one tokenization of the composed input."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib

import numpy as np
import torch

from .data import TokenGraphExample
from .features import summarize_attention_trace
from .trace import assign_segment_ids


def _unbatch(value):
    if isinstance(value, torch.Tensor) and value.ndim > 1:
        return value[0]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def tokenize_example_once(
    tokenizer,
    example: TokenGraphExample,
    *,
    max_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize the final passage/question/answer text exactly once."""

    encoded = tokenizer(
        example.text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
        truncation=False,
    )
    input_ids = torch.as_tensor(_unbatch(encoded["input_ids"]), dtype=torch.long)
    if max_tokens is not None and len(input_ids) > max_tokens:
        raise ValueError(
            f"example {example.example_id!r} has {len(input_ids)} tokens and "
            f"exceeds max_tokens={max_tokens}; full-context extraction refuses truncation"
        )
    offsets = torch.as_tensor(_unbatch(encoded["offset_mapping"]), dtype=torch.long)
    special = torch.as_tensor(
        _unbatch(encoded["special_tokens_mask"]), dtype=torch.bool
    )
    segment_ids = torch.tensor(
        assign_segment_ids(offsets.tolist(), example.segment_char_spans, special.tolist()),
        dtype=torch.long,
    )
    answer_mask = segment_ids == 3
    if not bool(answer_mask.any()):
        raise ValueError(f"answer segment for {example.example_id!r} has no tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": torch.as_tensor(
            _unbatch(encoded["attention_mask"]), dtype=torch.long
        ),
        "offset_mapping": offsets,
        "special_tokens_mask": special,
        "segment_ids": segment_ids,
        "answer_mask": answer_mask,
    }


def compute_teacher_forced_statistics(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align each next-token log probability and entropy to its target token."""

    values = logits[0] if logits.ndim == 3 else logits
    tokens = input_ids[0] if input_ids.ndim == 2 else input_ids
    if values.ndim != 2 or len(values) != len(tokens):
        raise ValueError("logits and input_ids must share their token dimension")
    token_log_prob = values.new_zeros(len(tokens), dtype=torch.float32)
    entropy = values.new_zeros(len(tokens), dtype=torch.float32)
    valid = torch.zeros(len(tokens), dtype=torch.bool, device=values.device)
    if len(tokens) < 2:
        return token_log_prob, entropy, valid
    distribution = torch.log_softmax(values[:-1].float(), dim=-1)
    token_log_prob[1:] = distribution.gather(1, tokens[1:].unsqueeze(1)).squeeze(1)
    entropy[1:] = -(distribution.exp() * distribution).sum(dim=-1)
    valid[1:] = True
    return token_log_prob, entropy, valid


def extract_example_trace(
    model,
    tokenizer,
    example: TokenGraphExample,
    *,
    selected_hidden_layers: Sequence[int],
    max_tokens: int | None = None,
) -> dict[str, object]:
    """Run one teacher-forced forward pass and return a label-free trace."""

    encoded = tokenize_example_once(tokenizer, example, max_tokens=max_tokens)
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")
    model_inputs = {
        "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
        "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
    }
    with torch.no_grad():
        outputs = model(
            **model_inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    if not outputs.attentions or any(value is None for value in outputs.attentions):
        raise RuntimeError(
            "The model did not return attention tensors; load it with eager attention"
        )
    attention = torch.stack(
        [value[0].detach().cpu().float() for value in outputs.attentions], dim=0
    )
    if not selected_hidden_layers:
        raise ValueError("at least one hidden-state layer must be selected")
    hidden_count = len(outputs.hidden_states)
    layer_ids = [index if index >= 0 else hidden_count + index for index in selected_hidden_layers]
    if any(index < 0 or index >= hidden_count for index in layer_ids):
        raise ValueError(
            f"selected hidden layers {tuple(selected_hidden_layers)} exceed "
            f"the available range for {hidden_count} hidden-state tensors"
        )
    hidden_states = torch.stack(
        [outputs.hidden_states[index][0].detach().cpu().float() for index in layer_ids],
        dim=0,
    )
    token_log_prob, entropy, valid = compute_teacher_forced_statistics(
        outputs.logits, model_inputs["input_ids"]
    )
    return {
        "schema_version": "token_trace_v1",
        "example_id": example.example_id,
        "pair_id": example.pair_id,
        "dataset": example.dataset,
        "text_sha256": hashlib.sha256(example.text.encode("utf-8")).hexdigest(),
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"].bool(),
        "offset_mapping": encoded["offset_mapping"],
        "special_tokens_mask": encoded["special_tokens_mask"],
        "segment_ids": encoded["segment_ids"],
        "answer_mask": encoded["answer_mask"],
        "attention": attention,
        "selected_hidden_layers": torch.tensor(layer_ids, dtype=torch.long),
        "hidden_states": hidden_states,
        "token_log_prob": token_log_prob.detach().cpu(),
        "next_token_entropy": entropy.detach().cpu(),
        "token_stat_valid": valid.detach().cpu(),
    }


def _token_spans(segment_ids: torch.Tensor) -> dict[str, tuple[int, int]]:
    spans = {}
    for name, identifier in (("passage", 1), ("question", 2), ("answer", 3)):
        indices = torch.nonzero(segment_ids == identifier, as_tuple=False).flatten()
        if not len(indices):
            raise ValueError(f"trace has no {name} tokens")
        spans[name] = (int(indices.min()), int(indices.max()) + 1)
    return spans


def _scalar(value) -> float:
    return float(np.asarray(value, dtype=float).mean())


def summarize_trace_record(trace: dict[str, object]) -> dict[str, object]:
    """Convert a raw trace to scalar observables before labels are available."""

    segment_ids = torch.as_tensor(trace["segment_ids"], dtype=torch.long)
    summary = summarize_attention_trace(
        trace["attention"], _token_spans(segment_ids)
    )
    record: dict[str, object] = {
        "example_id": str(trace["example_id"]),
        "pair_id": str(trace.get("pair_id", trace["example_id"])),
        "dataset": str(trace.get("dataset", "unknown")),
    }
    for name, value in summary.items():
        record[name] = _scalar(value)
    answer_mask = segment_ids == 3
    if "token_log_prob" in trace:
        valid = torch.as_tensor(
            trace.get("token_stat_valid", torch.ones_like(answer_mask)),
            dtype=torch.bool,
        )
        selected = answer_mask & valid
        if bool(selected.any()):
            values = torch.as_tensor(trace["token_log_prob"], dtype=torch.float32)
            record["mean_answer_log_prob"] = float(values[selected].mean())
    if "next_token_entropy" in trace:
        valid = torch.as_tensor(
            trace.get("token_stat_valid", torch.ones_like(answer_mask)),
            dtype=torch.bool,
        )
        selected = answer_mask & valid
        if bool(selected.any()):
            values = torch.as_tensor(trace["next_token_entropy"], dtype=torch.float32)
            record["mean_answer_next_token_entropy"] = float(values[selected].mean())
    return record
