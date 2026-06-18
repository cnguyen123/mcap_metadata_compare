import sys
from collections import defaultdict
from mcap.reader import make_reader

if len(sys.argv) != 2:
    print("Run program by passing file mcap name: mcap_meta_data_extract.py filename.mcap")
    sys.exit(1)

mcap_path = sys.argv[1]

topic_stats = defaultdict(lambda: {
    "count": 0,
    "first_log_time": None,
    "last_log_time": None,
    "schema_name": None,
    "schema_encoding": None,
    "message_encoding": None,
    "min_msg_size": None,
    "max_msg_size": None,
    "example_size": None,
})

with open(mcap_path, "rb") as f:
    reader = make_reader(f)

    for schema, channel, message in reader.iter_messages():
        topic = channel.topic
        stats = topic_stats[topic]

        stats["count"] += 1
        stats["schema_name"] = schema.name if schema else None
        stats["schema_encoding"] = schema.encoding if schema else None
        stats["message_encoding"] = channel.message_encoding

        log_time = message.log_time
        if stats["first_log_time"] is None or log_time < stats["first_log_time"]:
            stats["first_log_time"] = log_time
        if stats["last_log_time"] is None or log_time > stats["last_log_time"]:
            stats["last_log_time"] = log_time

        msg_size = len(message.data)
        stats["example_size"] = msg_size

        if stats["min_msg_size"] is None or msg_size < stats["min_msg_size"]:
            stats["min_msg_size"] = msg_size
        if stats["max_msg_size"] is None or msg_size > stats["max_msg_size"]:
            stats["max_msg_size"] = msg_size

with open("mcap_summary.txt", "w") as out:
    out.write(f"MCAP file: {mcap_path}\n")
    out.write(f"Number of topics: {len(topic_stats)}\n\n")

    for topic, stats in sorted(topic_stats.items()):
        first_s = stats["first_log_time"] / 1e9 if stats["first_log_time"] else None
        last_s = stats["last_log_time"] / 1e9 if stats["last_log_time"] else None
        duration = last_s - first_s if first_s is not None and last_s is not None else None
        rate = stats["count"] / duration if duration and duration > 0 else None

        out.write("=" * 80 + "\n")
        out.write(f"Topic: {topic}\n")
        out.write(f"Message count: {stats['count']}\n")
        out.write(f"Schema name: {stats['schema_name']}\n")
        out.write(f"Schema encoding: {stats['schema_encoding']}\n")
        out.write(f"Message encoding: {stats['message_encoding']}\n")
        out.write(f"First log time [s]: {first_s}\n")
        out.write(f"Last log time [s]: {last_s}\n")
        out.write(f"Duration [s]: {duration}\n")
        out.write(f"Estimated rate [Hz]: {rate}\n")
        out.write(f"Min message size [bytes]: {stats['min_msg_size']}\n")
        out.write(f"Max message size [bytes]: {stats['max_msg_size']}\n")
        out.write(f"Example message size [bytes]: {stats['example_size']}\n\n")

print("Wrote summary to mcap_summary.txt")