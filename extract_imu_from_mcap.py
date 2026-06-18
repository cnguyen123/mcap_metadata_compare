import argparse
import csv
import math
from collections import defaultdict

from mcap_ros2.reader import read_ros2_messages


def vec3_to_tuple(v):
    return float(v.x), float(v.y), float(v.z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mcap_file", help="Path to  .mcap file")
    parser.add_argument("--topic", default="/livox/imu", help="IMU topic name")
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="How many seconds to export. Use 0 for full file.")
    parser.add_argument("--raw-out", default="imu_sample.csv")
    parser.add_argument("--features-out", default="imu_1s_features.csv")
    args = parser.parse_args()

    rows = []
    start_time_ns = None

    for msg in read_ros2_messages(args.mcap_file, topics=[args.topic]):
        ros_msg = msg.ros_msg

        # Use MCAP log time as the primary timestamp.
        t_ns = msg.log_time_ns
        if start_time_ns is None:
            start_time_ns = t_ns

        rel_time_s = (t_ns - start_time_ns) / 1e9

        if args.max_seconds > 0 and rel_time_s > args.max_seconds:
            break

        wx, wy, wz = vec3_to_tuple(ros_msg.angular_velocity)
        ax, ay, az = vec3_to_tuple(ros_msg.linear_acceleration)

        gyro_mag = math.sqrt(wx * wx + wy * wy + wz * wz)
        acc_mag = math.sqrt(ax * ax + ay * ay + az * az)

        rows.append({
            "time_s": rel_time_s,
            "log_time_ns": t_ns,
            "wx": wx,
            "wy": wy,
            "wz": wz,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gyro_mag": gyro_mag,
            "acc_mag": acc_mag,
        })

    if not rows:
        print(f"No messages found for topic: {args.topic}")
        return

    #write raw IMU sample.
    with open(args.raw_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    #build 1-second motion fingerprint features.
    buckets = defaultdict(list)
    for r in rows:
        sec = int(r["time_s"])
        buckets[sec].append(r)

    feature_rows = []

    for sec in sorted(buckets.keys()):
        bucket = buckets[sec]

        gyro_vals = [r["gyro_mag"] for r in bucket]
        acc_vals = [r["acc_mag"] for r in bucket]

        def mean(xs):
            return sum(xs) / len(xs)

        def std(xs):
            m = mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

        feature_rows.append({
            "window_s": sec,
            "num_samples": len(bucket),

            "gyro_mean": mean(gyro_vals),
            "gyro_std": std(gyro_vals),
            "gyro_max": max(gyro_vals),
            "gyro_energy": sum(x * x for x in gyro_vals),

            "acc_mean": mean(acc_vals),
            "acc_std": std(acc_vals),
            "acc_max": max(acc_vals),
            "acc_energy": sum(x * x for x in acc_vals),

            "wx_mean": mean([r["wx"] for r in bucket]),
            "wy_mean": mean([r["wy"] for r in bucket]),
            "wz_mean": mean([r["wz"] for r in bucket]),

            "ax_mean": mean([r["ax"] for r in bucket]),
            "ay_mean": mean([r["ay"] for r in bucket]),
            "az_mean": mean([r["az"] for r in bucket]),
        })

    with open(args.features_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_rows[0].keys())
        writer.writeheader()
        writer.writerows(feature_rows)

    print(f"Wrote raw IMU sample to: {args.raw_out}")
    print(f"Wrote 1-second IMU features to: {args.features_out}")
    print(f"Exported {len(rows)} IMU messages over {rows[-1]['time_s']:.2f} seconds.")


if __name__ == "__main__":
    main()