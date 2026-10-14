# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""deepseek_v4 loss masks (skill/scripts/data/dsv4_mask.py): assistant pieces only, pieces join to the encoder output."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("dsv4_mask", ROOT / "skill" / "scripts" / "data" / "dsv4_mask.py")
dsv4_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dsv4_mask)

BOS, EOS, USER, ASSISTANT, THINK_END = "<B>", "<E>", "<U>", "<A>", "</t>"


class FakeEncoder:
    """The chat-mode format of encoding_dsv4: BOS, system text, <User>text<Assistant></think>, content<EOS>."""

    bos_token = BOS

    def render_message(self, index, messages, thinking_mode, drop_thinking=True):
        assert thinking_mode == "chat"
        message = messages[index]
        if message["role"] == "assistant":
            return message["content"] + EOS
        text = (USER if message["role"] == "user" else "") + message["content"]
        following = messages[index + 1]["role"] if index + 1 < len(messages) else None
        if message["role"] == "user" and following in (None, "assistant"):
            text += ASSISTANT + THINK_END
        return text

    def encode_messages(self, messages, thinking_mode):
        return BOS + "".join(self.render_message(i, messages, thinking_mode) for i in range(len(messages)))


class CharTokenizer:
    """One token per character, so every mask position maps back to a known character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


def _generator(encoder=None):
    return dsv4_mask.DeepSeekV4LossMaskGenerator(CharTokenizer(), encoder or FakeEncoder())


def _targets(tokens, mask):
    return "".join(chr(t) for t, m in zip(tokens, mask, strict=True) if m)


def test_only_assistant_content_and_eos_are_targets():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"}, {"role": "user", "content": "q"},
                {"role": "assistant", "content": "ans"}]
    tokens, mask = _generator().get_loss_mask(messages)
    assert "".join(chr(t) for t in tokens) == FakeEncoder().encode_messages(messages, "chat")
    assert _targets(tokens, mask) == "yo" + EOS + "ans" + EOS


def test_rejects_tools_unknown_roles_and_missing_assistant():
    gen = _generator()
    with pytest.raises(ValueError):
        gen.get_loss_mask([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                          tools=[{"name": "f"}])
    with pytest.raises(ValueError):
        gen.get_loss_mask([{"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}])
    with pytest.raises(ValueError):
        gen.get_loss_mask([{"role": "user", "content": "x"}])


def test_rendering_mismatch_is_refused():
    class Drifting(FakeEncoder):
        def encode_messages(self, messages, thinking_mode):
            return super().encode_messages(messages, thinking_mode) + "!"

    with pytest.raises(ValueError, match="differs"):
        _generator(Drifting()).get_loss_mask([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])


def test_load_encoder_needs_the_checkout_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        dsv4_mask.load_encoder(tmp_path)


@pytest.mark.skipif(not os.environ.get("ATTUNE_DSV4_TOKENIZER"), reason="set ATTUNE_DSV4_TOKENIZER to a DeepSeek-V4 "
                    "checkout with tokenizer.json and encoding/encoding_dsv4.py")
def test_real_encoder_and_tokenizer():
    from transformers import AutoTokenizer

    path = os.environ["ATTUNE_DSV4_TOKENIZER"]
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    gen = dsv4_mask.DeepSeekV4LossMaskGenerator(tokenizer, dsv4_mask.load_encoder(path))
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "6."}]
    tokens, mask = gen.get_loss_mask(messages)
    text = gen.encoder.encode_messages(messages, thinking_mode="chat")
    assert tokenizer.decode(tokens) == text
    assert tokens == tokenizer.encode(text, add_special_tokens=False)
    targets = tokenizer.decode([t for t, m in zip(tokens, mask, strict=True) if m])
    assert targets == "4." + gen.encoder.eos_token + "6." + gen.encoder.eos_token
    assert tokens[0] == tokenizer.convert_tokens_to_ids(gen.encoder.bos_token)
