# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assistant-only loss masks for DeepSeek-V4, whose tokenizer ships no chat template.

DeepSeek-V4 renders conversations with the official encoder `encoding/encoding_dsv4.py` in the model's Hugging Face
checkout. This generator renders each message in chat mode (`thinking_mode="chat"`), tokenizes the rendered pieces
in order, and marks the tokens of assistant pieces (content plus the end-of-sentence token) as targets. The pieces
must join to exactly `encode_messages(messages, thinking_mode="chat")`.
"""

import importlib.util
from pathlib import Path

ROLES = ("system", "user", "assistant")


def load_encoder(tokenizer_path: str | Path):
    """Import `encoding/encoding_dsv4.py` from the tokenizer checkout (Python code shipped with the model)."""
    path = Path(tokenizer_path) / "encoding" / "encoding_dsv4.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; DeepSeek-V4 masks need the checkout's official encoder")
    spec = importlib.util.spec_from_file_location("encoding_dsv4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepSeekV4LossMaskGenerator:
    def __init__(self, tokenizer, encoder):
        self.tokenizer = tokenizer
        self.encoder = encoder

    def render_pieces(self, messages: list[dict]) -> list[tuple[str, bool]]:
        """(text, is_target) per message; the BOS token opens the first piece."""
        for message in messages:
            if message.get("role") not in ROLES:
                raise ValueError(f"unsupported role {message.get('role')!r} (system, user and assistant only)")
            if not isinstance(message.get("content"), str):
                raise ValueError("message content must be a string")
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("conversation has no assistant turn")
        pieces = []
        for index, message in enumerate(messages):
            text = self.encoder.render_message(index, messages, thinking_mode="chat", drop_thinking=True)
            if index == 0:
                text = self.encoder.bos_token + text
            pieces.append((text, message["role"] == "assistant"))
        rendered = "".join(text for text, _ in pieces)
        expected = self.encoder.encode_messages(messages, thinking_mode="chat")
        if rendered != expected:
            raise ValueError("per-message rendering differs from encode_messages")
        return pieces

    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        if tools:
            raise ValueError("tool definitions are not supported by the deepseek_v4 mask")
        messages = [{"role": message.get("role"), "content": message.get("content")} for message in messages]
        tokens: list[int] = []
        mask: list[int] = []
        for text, is_target in self.render_pieces(messages):
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            tokens.extend(ids)
            mask.extend([1 if is_target else 0] * len(ids))
        return tokens, mask
