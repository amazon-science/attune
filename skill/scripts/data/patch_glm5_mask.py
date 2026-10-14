#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Idempotently add GLM-5 (glm5) SFT loss-mask support to slime.

slime's `mask_utils` has loss-mask generators for qwen, qwen3, qwen3_5 and distill_qwen (upstream main, checked
2026-10-11). GLM-5.x SFT needs `tokenizer_type="glm5"`. This patches the *resolved*
`slime.utils.mask_utils` module in place (works regardless of which slime on
PYTHONPATH wins) so the dataset generator and training both understand glm5.

Runs as a pre-step; a no-op if the installed slime already has glm5 support.
gen_multi_turn_loss_mask_glm5 is adapted from slime's gen_multi_turn_loss_mask_qwen3_5
(https://github.com/THUDM/slime, Apache-2.0; see THIRD_PARTY_NOTICES).

    python3 patch_glm5_mask.py      # takes no arguments
"""
import argparse
import importlib.util
import textwrap

METHOD = textwrap.dedent('''
    def gen_multi_turn_loss_mask_glm5(self, messages, tools=None):
        rendered_text = self.tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, add_generation_prompt=False)
        template_token_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, tools=tools, return_dict=False)
        stop_markers = ('<|endoftext|>', '<|user|>', '<|observation|>')
        if (messages and messages[-1]['role'] == 'assistant'
                and messages[-1].get('step_loss_mask', 1) == 1
                and not rendered_text.endswith(stop_markers)):
            rendered_text += '<|user|>'
        tokenized = self.tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids, offset_mapping = tokenized['input_ids'], tokenized.get('offset_mapping')
        if offset_mapping is None:
            raise ValueError('GLM-5 loss mask requires a fast tokenizer with return_offsets_mapping')
        if token_ids[:len(template_token_ids)] != template_token_ids:
            raise ValueError('GLM-5 rendered text tokenization does not match apply_chat_template(tokenize=True) output.')
        assistant_header = '<|assistant|>'
        think_prefix, no_think_prefix = '<think>', '</think>'
        char_mask, cursor = [0] * len(rendered_text), 0
        for message in messages:
            if message['role'] != 'assistant':
                continue
            header_pos = rendered_text.find(assistant_header, cursor)
            if header_pos < 0:
                raise ValueError('Failed to locate <|assistant|> in rendered GLM-5 chat template output.')
            content_start = header_pos + len(assistant_header)
            end_pos = len(rendered_text)
            matched_stop = None
            for marker in stop_markers:
                mp = rendered_text.find(marker, content_start)
                if 0 <= mp < end_pos:
                    end_pos = mp
                    matched_stop = marker
            cursor = end_pos
            if message.get('step_loss_mask', 1) != 1:
                continue
            mask_start = content_start
            while mask_start < end_pos and rendered_text[mask_start].isspace():
                mask_start += 1
            if rendered_text.startswith(think_prefix, mask_start):
                mask_start += len(think_prefix)
            elif rendered_text.startswith(no_think_prefix, mask_start):
                mask_start += len(no_think_prefix)
            for pos in range(mask_start, end_pos):
                char_mask[pos] = 1
            if matched_stop is not None:
                for pos in range(end_pos, end_pos + len(matched_stop)):
                    char_mask[pos] = 1
        psum = [0]
        for v in char_mask:
            psum.append(psum[-1] + v)
        loss_mask = [0 if e <= s else (1 if psum[e] - psum[s] > 0 else 0) for s, e in offset_mapping]
        return token_ids, loss_mask
''')


def patch(src):
    """mask_utils source with the glm5 generator added (raises SystemExit when the slime layout changed)."""
    anchor = '    def get_loss_mask(self, messages'
    if anchor not in src:
        raise SystemExit("anchor 'def get_loss_mask' not found; slime layout changed")
    src = src.replace(anchor, textwrap.indent(METHOD.strip(), '    ') + '\n\n' + anchor)
    old_else = '        else:\n            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")'
    if old_else not in src:
        raise SystemExit("could not find the tokenizer_type else-branch to extend")
    new_else = ('        elif self.tokenizer_type == "glm5":\n'
                '            return self.gen_multi_turn_loss_mask_glm5(messages, tools)\n' + old_else)
    return src.replace(old_else, new_else)


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    spec = importlib.util.find_spec("slime.utils.mask_utils")
    if spec is None or not spec.origin:
        raise SystemExit("cannot locate slime.utils.mask_utils on PYTHONPATH")
    with open(spec.origin) as f:
        src = f.read()
    if "glm5" in src:
        print(f"glm5 already present in {spec.origin}; no-op")
        return
    with open(spec.origin, "w") as f:
        f.write(patch(src))
    print(f"patched {spec.origin} with glm5 loss-mask support")


if __name__ == "__main__":
    main()
