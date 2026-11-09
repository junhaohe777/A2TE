"""Run the existing attribute-mask function and print a screenshot-ready table.

Copy this single file to the TBPS project root, then run:
    python test_mask.py
    python test_mask.py --case punctuation
    python test_mask.py --case truncation

No JSON input or second display script is required.
"""

import argparse
import subprocess
import textwrap
from pathlib import Path


TEXT_LENGTH = 77
CAPTIONS = {
    "split": "A man with short dark hair wears a white windbreaker, "
    "black sweatpants, and black sneakers. He carries a brown "
    "rucksack with long black straps and holds a yellow raincoat "
    "in his left hand." ,
    "punctuation": "A man wears a dark-blue jacket and light-gray trousers.",
    "original": (
        "A man with short dark hair wears a white windbreaker, "
    "black sweatpants, and black sneakers. He carries a brown "
    "rucksack with long black straps and holds a yellow raincoat "
    "in his left hand." 
    ),
}


def make_truncation_caption(tokenizer):
    base = "A man wears a white "
    for n in range(TEXT_LENGTH):
        for extra in range(4):
            prefix = "He walks. " * n + "Well " * extra + base
            if len(tokenizer.encode(prefix)) == TEXT_LENGTH - 3:
                if len(tokenizer.encode("windbreaker")) < 2:
                    raise ValueError("This tokenizer does not split windbreaker; choose another split word.")
                return prefix + "windbreaker and blue jeans."
    raise ValueError("Unable to construct a boundary example with this tokenizer.")


def display(caption, tokenizer, tokenize, get_attribute_mask, all_positions=False):
    ids = tokenize(caption, tokenizer, TEXT_LENGTH, truncate=True).detach().cpu().tolist()
    mask = get_attribute_mask(caption, tokenizer, TEXT_LENGTH, truncate=True).detach().cpu().tolist()
    if len(ids) != TEXT_LENGTH or len(mask) != TEXT_LENGTH:
        raise ValueError("Expected token IDs and attribute mask to both have length 77.")
    full_ids = tokenizer.encode(caption)
    full_count = len(full_ids) + 2
    truncated = full_count > TEXT_LENGTH
    eos_position = min(len(full_ids) + 1, TEXT_LENGTH - 1)
    if ids[0] != tokenizer.encoder["<|startoftext|>"] or ids[eos_position] != tokenizer.encoder["<|endoftext|>"]:
        raise ValueError("Actual special-token positions differ from the expected tokenize convention.")
    groups = {}
    for position in range(eos_position + 1):
        value = float(mask[position])
        if value < 0 or not value.is_integer():
            raise ValueError(f"Invalid group value at position {position}: {value}")
        group = int(value)
        if group:
            groups.setdefault(group, []).append(ids[position])
    group_text = {group: tokenizer.decode(values).strip() for group, values in groups.items()}
    positions = list(range(eos_position + 1))
    if truncated and not all_positions:
        positions = positions[-18:]
    lines = ["ATTRIBUTE-TO-BPE GROUPING", "=" * 88]
    lines.extend(textwrap.wrap("Caption: " + caption, width=88))
    lines += [f"Full tokens: {full_count}   Context length: {TEXT_LENGTH}   Truncated: {truncated}"]
    if positions[0] > 0:
        lines.append(f"Showing positions {positions[0]}-{eos_position}; earlier positions omitted.")
    lines += ["-" * 88, f"{'Pos':>3}  {'Token ID':>8}  {'BPE token':<25}  {'Group':>5}  Attribute text", "-" * 88]
    for position in positions:
        token_id = int(ids[position])
        group = int(mask[position])
        piece = "[SOS]" if position == 0 else "[EOS]" if position == eos_position else tokenizer.decoder[token_id]
        lines.append(f"{position:3d}  {token_id:8d}  {piece:<25}  {group:5d}  {group_text.get(group, '-')}")
    lines += ["-" * 88, "Positions are zero-based. Group 0: unselected. </w>: end of a word."]
    if truncated:
        lines.append("Attribute text is decoded from retained tokens only.")
        lines.extend(textwrap.wrap("Discarded text: " + tokenizer.decode(full_ids[eos_position - 1:]), width=88))
    special_errors = [p for p in [0, eos_position] + list(range(eos_position + 1, TEXT_LENGTH)) if mask[p] != 0]
    if special_errors:
        lines.append(f"CHECK: special/padding positions have nonzero groups: {special_errors}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=["split", "punctuation", "truncation", "original"], default="split")
    parser.add_argument("--all-positions", action="store_true", help="Show every non-padding position in the truncation case")
    args = parser.parse_args()
    # Use the project's actual tokenizer and mask; do not replace their logic.
    from utils.simple_tokenizer import SimpleTokenizer
    from datasets.bases import get_attribute_mask, tokenize
    tokenizer = SimpleTokenizer()
    caption = make_truncation_caption(tokenizer) if args.case == "truncation" else CAPTIONS[args.case]
    output = display(caption, tokenizer, tokenize, get_attribute_mask, args.all_positions)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if git.returncode == 0:
            output += "Code commit: " + git.stdout.strip() + "\n"
    except (OSError, subprocess.TimeoutExpired):
        pass
    destination = Path(f"mask_{args.case}.txt")
    destination.write_text(output, encoding="utf-8")
    print(output, end="")
    print(f"Saved display: {destination.resolve()}")


if __name__ == "__main__":
    main()
