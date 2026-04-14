import io
import sys
import types

import pytest
import torch

from conftest import REPO_ROOT, load_module_from_path


def install_fake_h5py_hierarchy(monkeypatch):
    fake_h5py = types.ModuleType("h5py")
    fake_h5py_hl = types.ModuleType("h5py._hl")
    fake_h5py_files = types.ModuleType("h5py._hl.files")
    fake_h5py_files.sys = sys

    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)
    monkeypatch.setitem(sys.modules, "h5py._hl", fake_h5py_hl)
    monkeypatch.setitem(sys.modules, "h5py._hl.files", fake_h5py_files)


def load_decifer_model_module(monkeypatch):
    install_fake_h5py_hierarchy(monkeypatch)
    return load_module_from_path(
        REPO_ROOT / "decifer" / "decifer_model.py",
        "test_decifer_model_module",
    )


def test_flash_attention_disables_dropout_in_eval(monkeypatch):
    module = load_decifer_model_module(monkeypatch)

    captured = []

    def fake_sdp(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        captured.append(dropout_p)
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdp)

    config = module.DeciferConfig(
        block_size=8,
        vocab_size=module.TOKENIZER.vocab_size,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.3,
        bias=False,
    )
    attention = module.CausalSelfAttention(config)
    attention.flash = True
    x = torch.randn(1, 4, 8)

    attention.eval()
    attention(x)
    assert captured[-1] == 0.0

    attention.train()
    attention(x)
    assert captured[-1] == pytest.approx(0.3)


def test_generate_and_print_uses_instance_tokenizer(monkeypatch):
    module = load_decifer_model_module(monkeypatch)

    model = module.Decifer(
        module.DeciferConfig(
            block_size=8,
            vocab_size=module.TOKENIZER.vocab_size,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
    )

    def fake_forward(idx, cond_vec=None, targets=None, start_indices_batch=None, custom_cond_emb=None):
        logits = torch.full((idx.size(0), 1, model.config.vocab_size), -1e9)
        logits[:, :, module.PADDING_ID] = 0.0
        return logits, None

    monkeypatch.setattr(model, "forward", fake_forward)

    buffer = io.StringIO()
    monkeypatch.setattr(module.sys, "stdout", buffer)

    start_id = module.TOKENIZER.token_to_id["data_"]
    prompt = torch.tensor([[start_id]], dtype=torch.long)
    model.generate_and_print(prompt, max_new_tokens=1)

    assert buffer.getvalue().startswith("data_")
