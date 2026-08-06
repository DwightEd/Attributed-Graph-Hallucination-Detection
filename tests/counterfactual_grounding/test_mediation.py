"""Gate 1 contracts for the CEPT Llama K/V mediation backend.

These tests define the RED phase before ``teacher.mediation`` exists.  The
backend is deliberately tested with a real, randomly initialised tiny Llama:
projection hooks or a fake attention module would not establish the promised
post-RoPE, pre-``repeat_kv`` intervention site.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from counterfactual_grounding.teacher.mediation import (
    LlamaKVMediationBackend,
    UnsupportedMediationBackend,
    decompose_mediation_effects,
    target_token_log_probs,
)


def _tiny_llama() -> LlamaForCausalLM:
    """Build a deterministic GQA Llama small enough for a CPU contract test."""

    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pad_token_id=0,
        use_cache=False,
    )
    return LlamaForCausalLM(config).cpu().eval()


def _teacher_forcing_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Positions 0--3 are prompt and positions 4--6 are response targets.
    input_ids = torch.tensor([[5, 7, 9, 11, 13, 15, 17]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    target_positions = torch.tensor([4, 5, 6], dtype=torch.long)
    return input_ids, attention_mask, target_positions


class MediationMathGate1Tests(unittest.TestCase):
    def test_target_log_probability_uses_t_minus_one_predictor(self):
        input_ids = torch.tensor([[1, 3, 2, 1, 0]], dtype=torch.long)
        target_positions = torch.tensor([2, 4], dtype=torch.long)
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [-2.0, -1.0, 4.0, 0.0],  # predicts token at position 2
                    [5.0, -3.0, -3.0, -3.0],  # off-by-one sentinel
                    [3.0, -2.0, -1.0, 0.0],  # predicts token at position 4
                    [-4.0, 5.0, -4.0, -4.0],  # off-by-one sentinel
                ]
            ],
            dtype=torch.float32,
        )

        observed = target_token_log_probs(logits, input_ids, target_positions)
        expected = torch.stack(
            (
                torch.log_softmax(logits[0, 1], dim=-1)[2],
                torch.log_softmax(logits[0, 3], dim=-1)[0],
            )
        ).unsqueeze(0)

        self.assertEqual(observed.shape, (1, 2))
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)

    def test_four_conditions_produce_registered_effects_and_exact_contract(self):
        y11 = torch.tensor([[-1.0, -2.0]], dtype=torch.float64)
        y00 = torch.tensor([[-3.0, -4.0]], dtype=torch.float64)
        y10 = torch.tensor([[-2.0, -2.5]], dtype=torch.float64)
        y01 = torch.tensor([[-2.4, -3.1]], dtype=torch.float64)

        effects = decompose_mediation_effects(
            y11=y11,
            y00=y00,
            y10=y10,
            y01=y01,
        )

        torch.testing.assert_close(effects.total, y11 - y00)
        torch.testing.assert_close(effects.direct, y10 - y00)
        torch.testing.assert_close(effects.mediated, y11 - y10)
        torch.testing.assert_close(effects.alternate_mediated, y01 - y00)
        torch.testing.assert_close(
            effects.interaction,
            effects.mediated - effects.alternate_mediated,
        )
        torch.testing.assert_close(
            effects.total,
            effects.direct + effects.mediated,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            effects.contract_residual,
            torch.zeros_like(effects.total),
            atol=0,
            rtol=0,
        )


class LlamaKVMediationBackendGate1Tests(unittest.TestCase):
    @unittest.skipUnless(
        transformers.__version__ == "4.52.3",
        "real K/V integration requires exactly transformers==4.52.3",
    )
    def test_capture_visits_every_layer_at_post_rope_pre_repeat_kv_site(self):
        model = _tiny_llama()
        backend = LlamaKVMediationBackend(model)
        input_ids, attention_mask, target_positions = _teacher_forcing_fixture()
        capture_positions = torch.tensor([4, 5, 6], dtype=torch.long)

        run = backend.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            capture_positions=capture_positions,
        )

        self.assertIsNotNone(run.kv)
        assert run.kv is not None
        self.assertEqual(set(run.kv.keys), {0, 1})
        self.assertEqual(set(run.kv.values), {0, 1})
        self.assertEqual(run.kv.positions.tolist(), [4, 5, 6])
        expected_shape = (1, 2, 3, 8)
        for layer_index in range(model.config.num_hidden_layers):
            self.assertEqual(tuple(run.kv.keys[layer_index].shape), expected_shape)
            self.assertEqual(tuple(run.kv.values[layer_index].shape), expected_shape)

        # Independently reconstruct layer-0 K/V.  Equality with rotated K and
        # unrotated V distinguishes this site from k_proj/v_proj hooks; two KV
        # heads instead of four query heads distinguishes it from post-repeat.
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        with torch.inference_mode():
            embeddings = model.model.embed_tokens(input_ids)
            normalized = model.model.layers[0].input_layernorm(embeddings)
            attention = model.model.layers[0].self_attn
            head_dim = attention.head_dim
            query = (
                attention.q_proj(normalized)
                .view(
                    1,
                    input_ids.shape[1],
                    model.config.num_attention_heads,
                    head_dim,
                )
                .transpose(1, 2)
            )
            raw_key = (
                attention.k_proj(normalized)
                .view(
                    1,
                    input_ids.shape[1],
                    model.config.num_key_value_heads,
                    head_dim,
                )
                .transpose(1, 2)
            )
            raw_value = (
                attention.v_proj(normalized)
                .view(
                    1,
                    input_ids.shape[1],
                    model.config.num_key_value_heads,
                    head_dim,
                )
                .transpose(1, 2)
            )
            cos, sin = model.model.rotary_emb(embeddings, position_ids)
            _, rotated_key = apply_rotary_pos_emb(query, raw_key, cos, sin)

        expected_key = rotated_key[:, :, capture_positions, :]
        expected_value = raw_value[:, :, capture_positions, :]
        self.assertFalse(
            torch.allclose(
                expected_key,
                raw_key[:, :, capture_positions, :],
                atol=1e-7,
                rtol=0,
            )
        )
        torch.testing.assert_close(run.kv.keys[0], expected_key, atol=1e-6, rtol=0)
        torch.testing.assert_close(run.kv.values[0], expected_value, atol=1e-6, rtol=0)

    @unittest.skipUnless(
        transformers.__version__ == "4.52.3",
        "real K/V integration requires exactly transformers==4.52.3",
    )
    def test_self_patch_preserves_all_target_log_probabilities(self):
        model = _tiny_llama()
        backend = LlamaKVMediationBackend(model)
        input_ids, attention_mask, target_positions = _teacher_forcing_fixture()
        response_positions = torch.tensor([4, 5, 6], dtype=torch.long)

        natural = backend.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            capture_positions=response_positions,
        )
        assert natural.kv is not None
        self_patched = backend.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            sender=natural.kv,
            patch_positions=response_positions,
        )

        torch.testing.assert_close(
            self_patched.target_log_probs,
            natural.target_log_probs,
            atol=1e-6,
            rtol=0,
        )

    @unittest.skipUnless(
        transformers.__version__ == "4.52.3",
        "real K/V integration requires exactly transformers==4.52.3",
    )
    def test_future_history_block_cannot_change_earlier_predictions(self):
        model = _tiny_llama()
        backend = LlamaKVMediationBackend(model)
        factual_ids, attention_mask, target_positions = _teacher_forcing_fixture()
        counterfactual_ids = factual_ids.clone()
        counterfactual_ids[0, 1:3] = torch.tensor([8, 10])
        response_positions = torch.tensor([4, 5, 6], dtype=torch.long)

        factual = backend.run(
            input_ids=factual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            capture_positions=response_positions,
        )
        assert factual.kv is not None
        counterfactual = backend.run(
            input_ids=counterfactual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
        )
        patched = backend.run(
            input_ids=counterfactual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            sender=factual.kv,
            patch_positions=torch.tensor([5], dtype=torch.long),
        )

        # Position 5 is not visible to predictors 3 and 4, which predict
        # targets 4 and 5.  This also detects a custom AttentionInterface that
        # forgot to register the eager causal-mask implementation.
        torch.testing.assert_close(
            patched.target_log_probs[:, :2],
            counterfactual.target_log_probs[:, :2],
            atol=1e-7,
            rtol=0,
        )
        self.assertGreater(
            float(
                (patched.target_log_probs[:, 2] - counterfactual.target_log_probs[:, 2])
                .abs()
                .max()
            ),
            1e-7,
        )

    @unittest.skipUnless(
        transformers.__version__ == "4.52.3",
        "real K/V integration requires exactly transformers==4.52.3",
    )
    def test_patch_rejects_wrong_key_or_value_shape_before_broadcasting(self):
        model = _tiny_llama()
        backend = LlamaKVMediationBackend(model)
        input_ids, attention_mask, target_positions = _teacher_forcing_fixture()
        response_positions = torch.tensor([4, 5, 6], dtype=torch.long)
        natural = backend.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            capture_positions=response_positions,
        )
        assert natural.kv is not None

        for field_name in ("keys", "values"):
            with self.subTest(field_name=field_name):
                malformed = dict(getattr(natural.kv, field_name))
                malformed[0] = malformed[0][..., :-1]
                bad_store = replace(natural.kv, **{field_name: malformed})

                with self.assertRaisesRegex(ValueError, "K/V|key|value|shape"):
                    backend.run(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        target_positions=target_positions,
                        sender=bad_store,
                        patch_positions=response_positions,
                    )

    def test_backend_rejects_transformers_versions_without_supported_patch_site(self):
        model = _tiny_llama()

        for version in ("4.52.2", "5.14.1", "6.0.0"):
            with (
                self.subTest(version=version),
                patch(
                    "counterfactual_grounding.teacher.mediation."
                    "importlib.metadata.version",
                    return_value=version,
                ),
                self.assertRaisesRegex(
                    UnsupportedMediationBackend,
                    "transformers|4\\.52\\.3|unsupported|exactly",
                ),
            ):
                LlamaKVMediationBackend(model)

        if transformers.__version__ != "4.52.3":
            with self.assertRaises(UnsupportedMediationBackend):
                LlamaKVMediationBackend(model)


if __name__ == "__main__":
    unittest.main()
