# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpt_oss loss masks (skill/scripts/data/patch_gptoss_mask.py): assistant turns only, date frozen, unfaithful turns refused."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("patch_gptoss_mask", ROOT / "skill" / "scripts" / "data" / "patch_gptoss_mask.py")
patch_gptoss_mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patch_gptoss_mask)

MASK_UTILS = '''
class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer, tokenizer_type="qwen"):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type

    def get_loss_mask(self, messages, tools=None):
        if self.tokenizer_type == "qwen":
            return None
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
'''


class HarmonyTokenizer:
    """A harmony-shaped template over one token per character, so mask positions map back to characters."""

    def apply_chat_template(self, messages, tokenize=False, return_dict=False, strftime_now=None, **kwargs):
        text = "<|start|>system<|message|>Current date: " + strftime_now("%Y-%m-%d") + "<|end|>"
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                end = "<|return|>" if i == len(messages) - 1 else "<|end|>"
                if m.get("thinking") and i == len(messages) - 1:
                    text += "<|start|>assistant<|channel|>analysis<|message|>" + m["thinking"] + "<|end|>"
                text += "<|start|>assistant<|channel|>final<|message|>" + m["content"] + end
            else:
                text += "<|start|>" + ("developer" if m["role"] == "system" else "user") + "<|message|>" + m["content"] + "<|end|>"
        return [ord(c) for c in text] if tokenize else text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def _generator():
    namespace = {}
    exec(patch_gptoss_mask.patch(MASK_UTILS), namespace)
    return namespace["MultiTurnLossMaskGenerator"](HarmonyTokenizer(), tokenizer_type="gpt_oss")


def _supervised(tokens, mask):
    return "".join(chr(t) for t, m in zip(tokens, mask) if m)


def test_assistant_turns_supervised_through_their_end_marker():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
                {"role": "user", "content": "more"}, {"role": "assistant", "content": "ok"}]
    tokens, mask = _generator().get_loss_mask(messages)
    assert len(tokens) == len(mask)
    assert _supervised(tokens, mask) == ("<|channel|>final<|message|>yo<|end|>" "<|channel|>final<|message|>ok<|return|>")
    assert "Current date: 2026-10-01" in "".join(chr(t) for t in tokens)


@pytest.mark.parametrize("messages", [
    [{"role": "user", "content": "a <|end|> b"}, {"role": "assistant", "content": "x"}],
    [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}, {"role": "system", "content": "s"},
     {"role": "assistant", "content": "y"}],
    [{"role": "user", "content": "q"}],
])
def test_unfaithful_conversations_refused(messages):
    with pytest.raises(ValueError):
        _generator().get_loss_mask(messages)


def test_system_turn_renders_as_unsupervised_developer_block():
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"}]
    tokens, mask = _generator().get_loss_mask(messages)
    text = "".join(chr(t) for t in tokens)
    assert "<|start|>developer<|message|>be brief<|end|>" in text
    assert _supervised(tokens, mask) == "<|channel|>final<|message|>yo<|return|>"


def test_last_turn_thinking_refused():
    # the template renders a final turn's thinking as a second assistant segment, which the mask cannot attribute
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "thinking": "t"}]
    with pytest.raises(ValueError):
        _generator().get_loss_mask(messages)


@pytest.mark.skipif(not os.environ.get("ATTUNE_GPTOSS_TOKENIZER"), reason="set ATTUNE_GPTOSS_TOKENIZER to a gpt-oss "
                    "checkout with tokenizer.json and chat_template.jinja")
def test_real_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["ATTUNE_GPTOSS_TOKENIZER"], use_fast=True)
    namespace = {}
    exec(patch_gptoss_mask.patch(MASK_UTILS), namespace)
    gen = namespace["MultiTurnLossMaskGenerator"](tokenizer, tokenizer_type="gpt_oss")
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."}, {"role": "user", "content": "And 3+3?"},
                {"role": "assistant", "content": "6."}]
    tokens, mask = gen.get_loss_mask(messages)
    assert len(tokens) == len(mask)
    assert "Current date: 2026-10-01" in tokenizer.decode(tokens)
    targets = tokenizer.decode([t for t, m in zip(tokens, mask, strict=True) if m])
    assert targets == "<|channel|>final<|message|>4.<|end|><|channel|>final<|message|>6.<|return|>"
