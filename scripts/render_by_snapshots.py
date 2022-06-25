import argparse
import os
import commentjson as json

import numpy as np

import sys
import time

from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf

from tqdm import tqdm

import pyngp as ngp # noqa

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene_dir", default="")
	parser.add_argument("--filename", default="transforms.json")
	parser.add_argument("--snapshots", nargs="*")
	parser.add_argument("--frames", nargs="*")
	parser.add_argument("--spp", default=16)
	parser.add_argument("--height", default=0)
	parser.add_argument("--width", default=0)
	parser.add_argument("--max", default=0)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	snapshotList = []
	if not args.snapshots:
		for filename in os.scandir(args.scene_dir + "/snapshots"):
			if filename.name.startswith('.') and filename.is_dir():
				continue
			snapshotList.append(filename.name)
	else:
		snapshotList = args.snapshots

	ref_transforms = {}
	with open(args.scene_dir + "/" + args.filename) as f:
		ref_transforms = json.load(f)

	frameList = {}
	if args.frames:
		for i in args.frames:
			for j in ref_transforms["frames"]:
				if j["file_path"].find("/" + str(i) + ".") >= 0:
					frameList[i] = j
	else:
		for j in ref_transforms["frames"]:
			frameList[os.path.basename(j["file_path"])] = j

	for snapshot in snapshotList:
		filepath = args.scene_dir + "/snapshots/" + snapshot
		name = snapshot.replace(".msgpack", "")

		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
		testbed = ngp.Testbed(mode)
		# testbed.background_color = [255., 255., 255., 1.0]
		testbed.nerf.sharpen = float(0)

		if not os.path.isfile(filepath):
			filepath = filepath + ".msgpack"
		print("Loading snapshot ", filepath)
		testbed.load_snapshot(filepath)

		testbed.fov_axis = 0
		testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi

		render_counter = 0
		if int(args.max) > 0:
			render_per = (len(frameList) // int(args.max))
		else:
			render_per = 1
		for screenshot_name in frameList:
			render_counter = render_counter + 1
			if render_counter % render_per != 0:
				continue

			f = frameList[screenshot_name]
			cam_matrix = f["transform_matrix"]
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1, :])
			os.makedirs(os.path.join(args.scene_dir, "screenshots"), exist_ok=True)
			os.makedirs(os.path.join(args.scene_dir, "screenshots", name), exist_ok=True)
			outname = os.path.join(args.scene_dir, "screenshots", name, screenshot_name)

			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".png"

			print(f"rendering {outname}")
			start_rendering_at = time.time()
			image = testbed.render(int(args.width) or int(ref_transforms["w"]), int(args.height) or int(ref_transforms["h"]),
								   args.spp, True)
			print(f"{outname} rendered, {time.time() - start_rendering_at} elapsed")
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)

		del testbed
