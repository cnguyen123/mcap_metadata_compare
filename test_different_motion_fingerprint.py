#!/usr/bin/env python3

import argparse
import json
import numpy as np

from motion_fingerprint_compare import (
    extract_fingerprint,
    standardize_pair,
    aligned_distance,
)


def make_variant(fp: np.ndarray, mode: str) -> np.ndarray:
    """
    Create an artificial motion fingerprint variant.

    Modes:
      same        -> identical fingerprint
      shifted     -> remove first N windows
      reversed    -> reverse temporal order
      shuffled    -> randomly shuffle time windows
      second_half -> compare first half against second half
    """
    if mode == "same":
        return fp.copy()

    if mode == "reversed":
        return fp[::-1].copy()

    if mode == "shuffled":
        rng = np.random.default_rng(seed=42)
        indices = np.arange(len(fp))
        rng.shuffle(indices)
        return fp[indices].copy()

    raise ValueError(f"Unsupported mode: {mode}")


def compare_fingerprints(fp_a, fp_b, max_shift_windows, threshold):
    a_std, b_std = standardize_pair(fp_a, fp_b)

    distance, best_shift, overlap = aligned_distance(
        a_std,
        b_std,
        max_shift_windows=max_shift_windows,
    )

    motion_similar = distance <= threshold

    return {
        "motion_distance": float(distance),
        "best_shift_windows": int(best_shift),
        "overlap_windows": int(overlap),
        "motion_threshold": threshold,
        "motion_similar": bool(motion_similar),
        "decision": (
            "RUN_PHASH_VERIFICATION"
            if motion_similar
            else "MOTION_DIFFERENT_SKIP_PHASH_USUALLY"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap_file")
    parser.add_argument("--topic", default="/livox/imu")
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=["same", "reversed", "shuffled", "first_vs_second_half"],
        default="reversed",
    )
    parser.add_argument("--max-shift-windows", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1.0)

    args = parser.parse_args()

    result = extract_fingerprint(
        mcap_file=args.mcap_file,
        topic=args.topic,
        window_s=args.window_s,
    )

    fp = result.fingerprint

    if args.mode == "first_vs_second_half":
        mid = len(fp) // 2
        fp_a = fp[:mid]
        fp_b = fp[mid:]
    else:
        fp_a = fp
        fp_b = make_variant(fp, args.mode)

    comparison = compare_fingerprints(
        fp_a=fp_a,
        fp_b=fp_b,
        max_shift_windows=args.max_shift_windows,
        threshold=args.threshold,
    )

    output = {
        "mcap_file": args.mcap_file,
        "topic": args.topic,
        "mode": args.mode,
        "num_windows_a": int(len(fp_a)),
        "num_windows_b": int(len(fp_b)),
        **comparison,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()