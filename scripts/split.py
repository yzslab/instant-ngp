import argparse
import json
import os.path


def parse_args():
	p = argparse.ArgumentParser()

	p.add_argument("--file", required=True)
	p.add_argument("--mode", default="each")
	p.add_argument("--each", default=3)
	p.add_argument("--ranges", nargs="*")

	return p.parse_args()


def parse_ranges(ranges):
	range_list = []
	for range in ranges:
		range_values = range.split("-")
		range_list.append([int(range_values[0]), int(range_values[1])])
	return range_list


args = parse_args()
args.each = int(args.each)
range_list = []


def is_file_no_match_each(frame_count, file_no):
	return 1 < file_no < frame_count and (file_no % args.each) == 0


def is_file_no_in_range(frame_count, file_no):
	for range_values in range_list:
		if range_values[0] <= file_no <= range_values[1]:
			return True
	return False


if args.mode == "each":
	is_test_frame = is_file_no_match_each
else:
	range_list = parse_ranges(args.ranges)
	is_test_frame = is_file_no_in_range

train_frames = []
test_frames = []
with open(args.file, "r") as f:
	transform = json.load(f)
	frame_count = len(transform["frames"])
	for frame in transform["frames"]:
		file_no = int(os.path.basename(frame["file_path"]).replace(".jpg", ""))

		if is_test_frame(frame_count, file_no):
			test_frames.append(frame)
		else:
			train_frames.append(frame)

dir = os.path.dirname(args.file)
with open(os.path.join(dir, "transforms_train.json"), "w") as f:
	transform["frames"] = train_frames
	json.dump(transform, f, indent=2)
with open(os.path.join(dir, "transforms_test.json"), "w") as f:
	transform["frames"] = test_frames
	json.dump(transform, f, indent=2)
with open(os.path.join(dir, "transforms_val.json"), "w") as f:
	transform["frames"] = test_frames
	json.dump(transform, f, indent=2)
