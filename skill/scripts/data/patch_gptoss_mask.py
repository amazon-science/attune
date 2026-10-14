#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotently add gpt-oss (harmony) SFT loss-mask support to slime.

slime's `mask_utils` has loss-mask generators for qwen, qwen3, qwen3_5 and distill_qwen. gpt-oss renders
conversations in the harmony format, so the dataset generator needs `tokenizer_type="gpt_oss"`. This patches the
*resolved* `slime.utils.mask_utils` module in place and is a no-op if a gpt_oss generator is already there.

The mask follows the qwen3_5 convention: the assistant header `<|start|>assistant` is context, and everything after it
through the closing `<|end|>`, `<|return|>` or `<|call|>` is supervised. The chat template's `strftime_now` is frozen
to FROZEN_DATE, so the rendered system message does not depend on the build date. A conversation the template would
not render faithfully (a system message after the first turn, a harmony special-token string in the content, or a
last turn that is not the assistant's) raises ValueError, which the generator counts as malformed and skips.
gen_multi_turn_loss_mask_gpt_oss is adapted from slime's gen_multi_turn_loss_mask_qwen3_5
(https://github.com/THUDM/slime, Apache-2.0; see THIRD_PARTY_NOTICES).

    python3 patch_gptoss_mask.py      # takes no arguments
"""
import argparse
import importlib.util
import textwrap

FROZEN_DATE = "2026-10-01"

METHOD = textwrap.dedent('''
    def gen_multi_turn_loss_mask_gpt_oss(self, messages, tools=None):
        import datetime
        import re
        frozen = datetime.datetime.strptime("%s", "%%Y-%%m-%%d")
        kwargs = {"strftime_now": lambda fmt: frozen.strftime(fmt)}
        if tools:
            kwargs["tools"] = tools
        for i, m in enumerate(messages):
            if m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
                raise ValueError("gpt-oss loss mask: unsupported role or non-string content")
            if m["role"] == "system" and i != 0:
                raise ValueError("gpt-oss loss mask: system message after the first turn")
            if re.search(r"<\\|[A-Za-z0-9_]+\\|>", m["content"]):
                raise ValueError("gpt-oss loss mask: special-token string in content")
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("gpt-oss loss mask: last turn is not the assistant's")
        rendered_text = self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        if "Current date: %s" not in rendered_text:
            raise ValueError("gpt-oss loss mask: the strftime_now override did not reach the chat template")
        template_token_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
        tokenized = self.tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids, offset_mapping = tokenized["input_ids"], tokenized.get("offset_mapping")
        if offset_mapping is None:
            raise ValueError("gpt-oss loss mask requires a fast tokenizer with return_offsets_mapping")
        if list(token_ids) != list(template_token_ids):
            raise ValueError("gpt-oss rendered text tokenization does not match apply_chat_template(tokenize=True)")
        header, end_markers = "<|start|>assistant", ("<|end|>", "<|return|>", "<|call|>")
        char_mask, cursor = [0] * len(rendered_text), 0
        for message in messages:
            if message["role"] != "assistant":
                continue
            header_pos = rendered_text.find(header, cursor)
            if header_pos < 0:
                raise ValueError("gpt-oss loss mask: assistant header not found")
            start = header_pos + len(header)
            ends = [(rendered_text.find(e, start), e) for e in end_markers]
            ends = [(p, e) for p, e in ends if p >= 0]
            if not ends:
                raise ValueError("gpt-oss loss mask: assistant end marker not found")
            pos, marker = min(ends)
            cursor = pos + len(marker)
            if message.get("step_loss_mask", 1) != 1:
                continue
            for p in range(start, cursor):
                char_mask[p] = 1
        if rendered_text.find(header, cursor) >= 0:
            raise ValueError("gpt-oss loss mask: more assistant segments than assistant messages")
        if not rendered_text.endswith("<|return|>"):
            raise ValueError("gpt-oss loss mask: rendered conversation does not end with <|return|>")
        psum = [0]
        for v in char_mask:
            psum.append(psum[-1] + v)
        loss_mask = [0 if e <= s else (1 if psum[e] - psum[s] > 0 else 0) for s, e in offset_mapping]
        return list(token_ids), loss_mask
''' % (FROZEN_DATE, FROZEN_DATE))


def patch(src):
    """mask_utils source with the gpt_oss generator added (raises SystemExit when the slime layout changed)."""
    anchor = '    def get_loss_mask(self, messages'
    if anchor not in src:
        raise SystemExit("anchor 'def get_loss_mask' not found; slime layout changed")
    src = src.replace(anchor, textwrap.indent(METHOD.strip(), '    ') + '\n\n' + anchor)
    old_else = '        else:\n            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")'
    if old_else not in src:
        raise SystemExit("could not find the tokenizer_type else-branch to extend")
    new_else = ('        elif self.tokenizer_type == "gpt_oss":\n'
                '            return self.gen_multi_turn_loss_mask_gpt_oss(messages, tools)\n' + old_else)
    return src.replace(old_else, new_else)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    spec = importlib.util.find_spec("slime.utils.mask_utils")
    if spec is None or not spec.origin:
        raise SystemExit("cannot locate slime.utils.mask_utils on PYTHONPATH")
    with open(spec.origin) as f:
        src = f.read()
    if "gen_multi_turn_loss_mask_gpt_oss" in src:
        print(f"gpt_oss already present in {spec.origin}; no-op")
        return
    with open(spec.origin, "w") as f:
        f.write(patch(src))
    print(f"patched {spec.origin} with gpt_oss loss-mask support")


if __name__ == "__main__":
    main()
