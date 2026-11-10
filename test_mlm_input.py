import random
from textwrap import fill
from unittest.mock import patch

import numpy as np


CAPTION = (
    "A man with short dark hair wears a white windbreaker, "
    "black sweatpants, and black sneakers. He carries a brown "
    "rucksack with long black straps and holds a yellow raincoat "
    "in his left hand."
)
SEED = 1


def changed_positions(reference, sequence):
    return (sequence != reference).nonzero(as_tuple=True)[0].tolist()


def trace_loader(bases, dataset, copy_mlm_input):
    tokenize = bases.tokenize
    build_mlm = dataset._build_random_masked_tokens_and_labels
    caption_tensor = original = after_mlm = None
    shares_storage = None

    def observe_tokenize(*args, **kwargs):
        nonlocal caption_tensor, original
        tokens = tokenize(*args, **kwargs)
        if caption_tensor is None:
            caption_tensor = tokens
            original = tokens.clone()
        return tokens

    def observe_mlm(tokens):
        nonlocal after_mlm, shares_storage
        if caption_tensor is None:
            raise RuntimeError("The loader called MLM before tokenizing the caption.")
        mask_input = tokens.copy() if copy_mlm_input else tokens
        shares_storage = np.shares_memory(caption_tensor.cpu().numpy(), mask_input)
        result = build_mlm(mask_input)
        after_mlm = caption_tensor.clone()
        return result

    # Replay the repository's masking routine with the same random seed for each run.
    with patch.object(bases, "random", random.Random(SEED)), patch.object(
        bases, "read_image", return_value=None
    ), patch.object(
        bases, "tokenize", side_effect=observe_tokenize
    ), patch.object(dataset, "_build_random_masked_tokens_and_labels", side_effect=observe_mlm):
        sample = dataset[0]

    if original is None or after_mlm is None:
        raise RuntimeError("Could not observe the tokenization and MLM steps in this loader.")
    return original, after_mlm, sample, shares_storage


def main():
    from datasets import bases

    dataset = bases.ImageTextMLMDataset([(0, 0, "unused.jpg", CAPTION)])
    original, after_mlm, actual, actual_shares = trace_loader(bases, dataset, copy_mlm_input=False)
    control_original, control_after, control, control_shares = trace_loader(
        bases, dataset, copy_mlm_input=True
    )
    if not original.equal(control_original):
        raise RuntimeError("The two runs did not start from identical tokenized captions.")

    main_changes = changed_positions(original, actual["caption_ids"])
    control_changes = changed_positions(original, control["caption_ids"])
    mlm_changes = changed_positions(original, actual["mlm_ids"])
    same_mlm_inputs = actual["mlm_ids"].equal(control["mlm_ids"])
    same_mlm_targets = actual["mlm_labels"].equal(control["mlm_labels"])
    checked_targets = actual["mlm_labels"].ne(0) | actual["mlm_ids"].ne(original)
    targets_match_original = actual["mlm_labels"][checked_targets].equal(original[checked_targets])

    print(f"Implementation: {bases.__file__}\n")
    print("MLM caption preservation check")
    print(fill(CAPTION, width=78, initial_indent="Caption: ", subsequent_indent="         "))
    print(f"Sequence length: {original.numel()} token positions | Random seed: {SEED}")
    if not mlm_changes:
        print("No token replacement observed. Change SEED and rerun the check.")
        return 2

    position = (main_changes or mlm_changes)[0]
    tokenizer = dataset.tokenizer
    mask_id = tokenizer.encoder["<|mask|>"]

    def token_at(sequence):
        token_id = sequence[position].item()
        token = "[MASK]" if token_id == mask_id else tokenizer.decoder[token_id]
        return f"{token_id} ({token})"

    print(f"\nExample position: {position} (zero-based)")
    print(f"{'Observation':<29} {'Actual loader':>17} {'Changed':>8} {'Copy control':>17} {'Changed':>8}")
    for stage, actual_ids, control_ids in (
        ("Original after tokenization", original, control_original),
        ("Same tensor after MLM", after_mlm, control_after),
        ("Returned caption_ids", actual["caption_ids"], control["caption_ids"]),
        ("Returned mlm_ids", actual["mlm_ids"], control["mlm_ids"]),
    ):
        actual_count = len(changed_positions(original, actual_ids))
        control_count = len(changed_positions(original, control_ids))
        print(f"{stage:<29} {token_at(actual_ids):>17} {actual_count:>8} "
              f"{token_at(control_ids):>17} {control_count:>8}")

    print("\nChanged IDs are counted against the full original sequence.")
    print(f"MLM input shares caption storage (actual / copy): {actual_shares} / {control_shares}")
    print(f"caption_ids preserve the original (actual / copy): {not main_changes} / {not control_changes}")
    print(f"MLM target ID at example position (actual / copy): "
          f"{actual['mlm_labels'][position].item()} / {control['mlm_labels'][position].item()}")
    print(f"Full MLM input sequences equal across runs: {same_mlm_inputs}")
    print(f"Full MLM target sequences equal across runs: {same_mlm_targets}")
    print(f"MLM targets match original IDs at supervised or changed positions: {targets_match_original}")
    if not (same_mlm_inputs and same_mlm_targets):
        print("INCONCLUSIVE: the two runs did not produce the same MLM corruption and targets.")
        return 2
    return int(bool(main_changes or control_changes) or not targets_match_original)


if __name__ == "__main__":
    raise SystemExit(main())
