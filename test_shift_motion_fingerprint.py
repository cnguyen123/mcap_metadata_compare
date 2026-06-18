#!/usr/bin/env python3

import argparse
import json
import numpy as np

from motion_fingerprint_compare import (
    extract_fingerprint,
    standardize_pair,
    aligned_distance,
)


def shift_fingerprint(fp: np.ndarray, shift_windows: int) -> np.ndarray:
    """
    Create an artificial time-shifted version of a fingerprint.

    Positive shift:
        shifted version starts later, so we remove the first K windows.

    Example:
        original: [0, 1, 2, 3, 4, 5]
        shift=2:  [2, 3, 4, 5]
    """
    if shift_windows <= 0:
        return fp.copy()

    if shift_windows >= len(fp):
        raise ValueError("shift_windows is too large for this fingerprint.")

    return fp[shift_windows:].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap_file")
    parser.add_argument("--topic", default="/livox/imu")
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--shift-windows", type=int, default=2)
    parser.add_argument("--max-shift-windows", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1.0)

    args = parser.parse_args()

    result = extract_fingerprint(
        mcap_file=args.mcap_file,
        topic=args.topic,
        window_s=args.window_s,
    )

    original_fp = result.fingerprint
    shifted_fp = shift_fingerprint(original_fp, args.shift_windows)

    original_std, shifted_std = standardize_pair(original_fp, shifted_fp)

    distance, best_shift, overlap = aligned_distance(
        original_std,
        shifted_std,
        max_shift_windows=args.max_shift_windows,
    )

    motion_similar = distance <= args.threshold

    output = {
        "mcap_file": args.mcap_file,
        "topic": args.topic,
        "window_s": args.window_s,
        "num_original_windows": int(original_fp.shape[0]),
        "num_shifted_windows": int(shifted_fp.shape[0]),
        "artificial_shift_windows": args.shift_windows,
        "max_shift_windows_allowed": args.max_shift_windows,
        "best_shift_found": int(best_shift),
        "overlap_windows": int(overlap),
        "motion_distance": float(distance),
        "threshold": args.threshold,
        "motion_similar": bool(motion_similar),
        "decision": "SHIFT_HANDLED" if motion_similar else "SHIFT_NOT_HANDLED",
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()