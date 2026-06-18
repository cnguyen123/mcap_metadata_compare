#!/usr/bin/env python3
"""
Motion fingerprint comparison for two MCAP files.

Purpose
-------
the program performs the first-stage duplicate check for two MCAP files. In this program, we use the IMU topic, e.g. /livox/imu, to build a compact motion fingerprint
and decides whether the two MCAP files should be passed to visual pHash verification. [Can be extended to use other topic that is more informatic in the mcap file metadata]

Note that we do NOT delete files and do NOT make the final duplicate decision.
The intended policy is:

    motion similar  -> run pHash verification
    motion different -> usually skip pHash, or use optional cheap visual fallback

Install
-------
    pip install mcap-ros2-support numpy

Example
-------
    python motion_fingerprint_compare.py file_a.mcap file_b.mcap --topic /livox/imu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from mcap_ros2.reader import read_ros2_messages


@dataclass
class ImuSample:
    time_s: float
    wx: float
    wy: float
    wz: float
    ax: float
    ay: float
    az: float


@dataclass
class FingerprintResult:
    mcap_file: str
    topic: str
    duration_s: float
    num_samples: int
    num_windows: int
    feature_names: List[str]
    fingerprint: np.ndarray


def _get_log_time_ns(decoded_msg) -> int:
    if hasattr(decoded_msg, "log_time_ns"):
        return int(decoded_msg.log_time_ns)
    if hasattr(decoded_msg, "message") and hasattr(decoded_msg.message, "log_time"):
        return int(decoded_msg.message.log_time)
    if hasattr(decoded_msg, "log_time"):
        return int(decoded_msg.log_time)
    raise AttributeError("Cannot find log time on decoded MCAP message.")


def extract_imu_samples(mcap_file: str, topic: str) -> List[ImuSample]:
    samples: List[ImuSample] = []
    start_ns: Optional[int] = None

    for decoded in read_ros2_messages(mcap_file, topics=[topic]):
        ros_msg = decoded.ros_msg
        t_ns = _get_log_time_ns(decoded)

        if start_ns is None:
            start_ns = t_ns

        t = (t_ns - start_ns) / 1e9

        av = ros_msg.angular_velocity
        la = ros_msg.linear_acceleration

        samples.append(
            ImuSample(
                time_s=float(t),
                wx=float(av.x),
                wy=float(av.y),
                wz=float(av.z),
                ax=float(la.x),
                ay=float(la.y),
                az=float(la.z),
            )
        )

    return samples


def _safe_stats(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.max(values)),
        float(np.min(values)),
        float(np.sum(values * values)),
    )


def build_motion_fingerprint(
    samples: List[ImuSample],
    window_s: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    if not samples:
        raise ValueError("No IMU samples were found. Check the topic name.")

    arr = np.array(
        [[s.time_s, s.wx, s.wy, s.wz, s.ax, s.ay, s.az] for s in samples],
        dtype=np.float64,
    )

    t = arr[:, 0]
    wx, wy, wz = arr[:, 1], arr[:, 2], arr[:, 3]
    ax, ay, az = arr[:, 4], arr[:, 5], arr[:, 6]

    gyro_mag = np.sqrt(wx * wx + wy * wy + wz * wz)
    acc_mag = np.sqrt(ax * ax + ay * ay + az * az)

    num_windows = max(1, int(math.ceil((t[-1] + 1e-9) / window_s)))

    feature_names = [
        "gyro_mean",
        "gyro_std",
        "gyro_max",
        "gyro_min",
        "gyro_energy",
        "acc_mean",
        "acc_std",
        "acc_max",
        "acc_min",
        "acc_energy",
        "wx_mean",
        "wy_mean",
        "wz_mean",
        "wx_std",
        "wy_std",
        "wz_std",
        "ax_mean",
        "ay_mean",
        "az_mean",
        "ax_std",
        "ay_std",
        "az_std",
    ]

    rows = []

    for k in range(num_windows):
        start = k * window_s
        end = start + window_s
        mask = (t >= start) & (t < end)

        if not np.any(mask):
            rows.append([0.0] * len(feature_names))
            continue

        gm = gyro_mag[mask]
        am = acc_mag[mask]

        gyro_mean, gyro_std, gyro_max, gyro_min, gyro_energy = _safe_stats(gm)
        acc_mean, acc_std, acc_max, acc_min, acc_energy = _safe_stats(am)

        row = [
            gyro_mean,
            gyro_std,
            gyro_max,
            gyro_min,
            gyro_energy,
            acc_mean,
            acc_std,
            acc_max,
            acc_min,
            acc_energy,
            float(np.mean(wx[mask])),
            float(np.mean(wy[mask])),
            float(np.mean(wz[mask])),
            float(np.std(wx[mask])),
            float(np.std(wy[mask])),
            float(np.std(wz[mask])),
            float(np.mean(ax[mask])),
            float(np.mean(ay[mask])),
            float(np.mean(az[mask])),
            float(np.std(ax[mask])),
            float(np.std(ay[mask])),
            float(np.std(az[mask])),
        ]

        rows.append(row)

    return np.asarray(rows, dtype=np.float64), feature_names


def standardize_pair(fp_a: np.ndarray, fp_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([fp_a, fp_b])
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    std[std < 1e-9] = 1.0
    return (fp_a - mean) / std, (fp_b - mean) / std


def aligned_distance(
    fp_a: np.ndarray,
    fp_b: np.ndarray,
    max_shift_windows: int = 2,
) -> Tuple[float, int, int]:
    best_distance = float("inf")
    best_shift = 0
    best_overlap = 0

    for shift in range(-max_shift_windows, max_shift_windows + 1):
        if shift >= 0:
            a_part = fp_a[: max(0, min(len(fp_a), len(fp_b) - shift))]
            b_part = fp_b[shift : shift + len(a_part)]
        else:
            offset = -shift
            b_part = fp_b[: max(0, min(len(fp_b), len(fp_a) - offset))]
            a_part = fp_a[offset : offset + len(b_part)]

        overlap = len(a_part)
        if overlap <= 0:
            continue

        per_window = np.linalg.norm(a_part - b_part, axis=1)
        distance = float(np.mean(per_window))

        if distance < best_distance:
            best_distance = distance
            best_shift = shift
            best_overlap = overlap

    return best_distance, best_shift, best_overlap


def extract_fingerprint(
    mcap_file: str,
    topic: str,
    window_s: float,
) -> FingerprintResult:
    samples = extract_imu_samples(mcap_file, topic)

    if len(samples) < 2:
        raise ValueError(f"Not enough IMU samples found in {mcap_file} on topic {topic}")

    fp, feature_names = build_motion_fingerprint(samples, window_s=window_s)
    duration_s = samples[-1].time_s - samples[0].time_s

    return FingerprintResult(
        mcap_file=mcap_file,
        topic=topic,
        duration_s=float(duration_s),
        num_samples=len(samples),
        num_windows=fp.shape[0],
        feature_names=feature_names,
        fingerprint=fp,
    )


def save_fingerprint_csv(result: FingerprintResult, out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_index"] + result.feature_names)
        for i, row in enumerate(result.fingerprint):
            writer.writerow([i] + [float(x) for x in row])


def compare_two_mcaps(
    mcap_a: str,
    mcap_b: str,
    topic: str = "/livox/imu",
    window_s: float = 1.0,
    max_shift_windows: int = 2,
    duration_tolerance_s: float = 2.0,
    threshold: float = 1.0,
    export_features: bool = False,
) -> dict:
    a = extract_fingerprint(mcap_a, topic, window_s)
    b = extract_fingerprint(mcap_b, topic, window_s)

    duration_diff = abs(a.duration_s - b.duration_s)
    duration_similar = duration_diff <= duration_tolerance_s

    a_std, b_std = standardize_pair(a.fingerprint, b.fingerprint)

    distance, best_shift, overlap = aligned_distance(
        a_std,
        b_std,
        max_shift_windows=max_shift_windows,
    )

    motion_similar = distance <= threshold
    candidate_for_phash = duration_similar and motion_similar

    if export_features:
        save_fingerprint_csv(a, Path(mcap_a).stem + "_motion_fingerprint.csv")
        save_fingerprint_csv(b, Path(mcap_b).stem + "_motion_fingerprint.csv")

    return {
        "mcap_a": mcap_a,
        "mcap_b": mcap_b,
        "topic": topic,
        "window_s": window_s,
        "duration_a_s": a.duration_s,
        "duration_b_s": b.duration_s,
        "duration_diff_s": duration_diff,
        "duration_tolerance_s": duration_tolerance_s,
        "duration_similar": duration_similar,
        "imu_samples_a": a.num_samples,
        "imu_samples_b": b.num_samples,
        "windows_a": a.num_windows,
        "windows_b": b.num_windows,
        "max_shift_windows": max_shift_windows,
        "best_shift_windows": best_shift,
        "overlap_windows": overlap,
        "motion_distance": distance,
        "motion_threshold": threshold,
        "motion_similar": motion_similar,
        "candidate_for_phash": candidate_for_phash,
        "decision": (
            "RUN_PHASH_VERIFICATION"
            if candidate_for_phash
            else "SKIP_PHASH_USUALLY_OR_USE_CHEAP_VISUAL_FALLBACK"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two MCAP files using IMU motion fingerprints."
    )

    parser.add_argument("mcap_a", help="First MCAP file")
    parser.add_argument("mcap_b", help="Second MCAP file")
    parser.add_argument("--topic", default="/livox/imu", help="IMU topic name")
    parser.add_argument("--window-s", type=float, default=1.0, help="Window size in seconds")
    parser.add_argument(
        "--max-shift-windows",
        type=int,
        default=2,
        help="Allow +/- this many windows of time shift",
    )
    parser.add_argument(
        "--duration-tolerance-s",
        type=float,
        default=2.0,
        help="Maximum allowed duration difference before rejecting pair",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Motion distance threshold. Lower is stricter.",
    )
    parser.add_argument(
        "--export-features",
        action="store_true",
        help="Save per-file motion fingerprint CSV files",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save JSON result",
    )

    args = parser.parse_args()

    try:
        result = compare_two_mcaps(
            mcap_a=args.mcap_a,
            mcap_b=args.mcap_b,
            topic=args.topic,
            window_s=args.window_s,
            max_shift_windows=args.max_shift_windows,
            duration_tolerance_s=args.duration_tolerance_s,
            threshold=args.threshold,
            export_features=args.export_features,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()