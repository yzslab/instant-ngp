#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time
import datetime

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
	parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
	parser.add_argument("--name", default="")
	parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

	parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
	parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")
	parser.add_argument("--save_per_n", default=0)

	parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
	parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
	parser.add_argument("--test_max", default=0)
	parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
	parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

	parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
	parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
	parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
	parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

	parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
	parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
	parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
	parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
	parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
	parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video.")

	parser.add_argument("--max_time", default=0)
	parser.add_argument("--time_offset", default=0)

	parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

	parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
	parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

	parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
	parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")

	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")

	parser.add_argument("--cone_angle_constant", default=-1)

	parser.add_argument("--train_extrinsics", action="store_true", help="Enable extrinsics optimizer")
	parser.add_argument("--train_exposure", action="store_true", help="Enable exposure optimizer")
	parser.add_argument("--train_distortion", action="store_true", help="Enable distortion optimizer")
	parser.add_argument("--train_focal_length", action="store_true", help="Enable focal length optimizer")
	parser.add_argument("--train_envmap", action="store_true", help="Enable train envmap")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	args.mode = args.mode or mode_from_scene(args.scene) or mode_from_scene(args.load_snapshot)
	if not args.mode:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	if args.mode == "sdf":
		mode = ngp.TestbedMode.Sdf
		configs_dir = os.path.join(ROOT_DIR, "configs", "sdf")
		scenes = scenes_sdf
	elif args.mode == "nerf":
		mode = ngp.TestbedMode.Nerf
		configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
		scenes = scenes_nerf
	elif args.mode == "image":
		mode = ngp.TestbedMode.Image
		configs_dir = os.path.join(ROOT_DIR, "configs", "image")
		scenes = scenes_image
	elif args.mode == "volume":
		mode = ngp.TestbedMode.Volume
		configs_dir = os.path.join(ROOT_DIR, "configs", "volume")
		scenes = scenes_volume
	else:
		raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")

	base_network = os.path.join(configs_dir, "base.json")
	if args.scene in scenes:
		network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
		base_network = os.path.join(configs_dir, network+".json")
	network = args.network if args.network else base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)
	testbed.exposure = args.exposure
	if mode == ngp.TestbedMode.Sdf:
		testbed.tonemap_curve = ngp.TonemapCurve.ACES

	if args.scene:
		scene = args.scene
		if not os.path.exists(args.scene) and args.scene in scenes:
			scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
		testbed.load_training_data(scene)

	if args.gui:
		# Pick a sensible GUI resolution depending on arguments.
		sw = args.width or 1920
		sh = args.height or 1080
		while sw*sh > 1920*1080*4:
			sw = int(sw / 2)
			sh = int(sh / 2)
		testbed.init_window(sw, sh, second_window = args.second_window or False)


	if args.load_snapshot:
		if os.path.exists(args.load_snapshot) == False:
			args.load_snapshot = os.path.join(os.path.dirname(args.scene), "snapshots", args.name, args.load_snapshot)
		if os.path.exists(args.load_snapshot) == False:
			args.load_snapshot = args.load_snapshot + ".msgpack"

		snapshot = args.load_snapshot
		if not os.path.exists(snapshot) and snapshot in scenes:
			snapshot = default_snapshot_filename(scenes[snapshot])

		print("Loading snapshot ", snapshot)
		testbed.load_snapshot(snapshot)
	else:
		testbed.reload_network_from_file(network)

	ref_transforms = {}
	if args.screenshot_transforms: # try to load the given file straight away
		print("Screenshot transforms from ", args.screenshot_transforms)
		with open(args.screenshot_transforms) as f:
			ref_transforms = json.load(f)

	testbed.shall_train = args.train if args.gui else True


	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]
	if args.mode == "sdf":
		setup_colored_sdf(testbed, args.scene)

	if args.near_distance >= 0.0:
		print("NeRF training ray near_distance ", args.near_distance)
		testbed.nerf.training.near_distance = args.near_distance
	if args.train_extrinsics:
		print("Train extrinsics")
		testbed.nerf.training.optimize_extrinsics = True
	if args.train_exposure:
		print("Train exposure")
		testbed.nerf.training.optimize_exposure = True
	if args.train_distortion:
		print("Train distortion")
		testbed.nerf.training.optimize_distortion = True
	if args.train_focal_length:
		print("Train focal length")
		testbed.nerf.training.optimize_focal_length = True
	if args.train_envmap:
		print("Train envmap")
		testbed.nerf.training.train_envmap = True
	cone_angle_constant = float(args.cone_angle_constant)
	if cone_angle_constant >= 0:
		print(f"cone_angle_constant={args.cone_angle_constant}")
		testbed.nerf.cone_angle_constant = cone_angle_constant

	if args.nerf_compatibility:
		print(f"NeRF compatibility mode enabled")

		# Prior nerf papers accumulate/blend in the sRGB
		# color space. This messes not only with background
		# alpha, but also with DOF effects and the likes.
		# We support this behavior, but we only enable it
		# for the case of synthetic nerf data where we need
		# to compare PSNR numbers to results of prior work.
		testbed.color_space = ngp.ColorSpace.SRGB

		# No exponential cone tracing. Slightly increases
		# quality at the cost of speed. This is done by
		# default on scenes with AABB 1 (like the synthetic
		# ones), but not on larger scenes. So force the
		# setting here.
		testbed.nerf.cone_angle_constant = 0

		# Optionally match nerf paper behaviour and train on a
		# fixed white bg. We prefer training on random BG colors.
		# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		# testbed.nerf.training.random_bg_color = False

	old_training_step = 0
	n_steps = args.n_steps


	timepoints = [7200, 86400]
	timepoint_count = len(timepoints)
	current_timepoint = 0

	time_offset = int(args.time_offset)
	if time_offset > 0:
		while (current_timepoint < timepoint_count):
			if timepoints[current_timepoint] > time_offset:
				break
			current_timepoint += 1

	print(f"target timepoint: timepoints[{current_timepoint}]={timepoints[current_timepoint]}")

	if args.save_snapshot == "1":
		args.save_snapshot = os.path.join(os.path.dirname(args.scene), "snapshots", args.name)
		os.makedirs(args.save_snapshot, exist_ok=True)

	args.save_per_n = int(args.save_per_n)

	args.max_time = int(args.max_time)

	snapshot_save_time = 0

	# If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
	# don't train by default and instead assume that the goal is to render screenshots,
	# compute PSNR, or render a video.
	if n_steps < 0 and (not args.load_snapshot or args.gui):
		n_steps = 35000

	tqdm_last_update = 0

	trained_steps = testbed.training_step

	def save_snapshot(name):
		save_path = args.save_snapshot + "/" + str(name) + ".msgpack"
		print(f"Saving snapshot to ", save_path)
		snapshot_save_started_at = time.time()
		testbed.save_snapshot(save_path, False)
		return time.time() - snapshot_save_started_at

	if n_steps > 0:
		log_directory = os.path.join(os.path.dirname(args.scene), "logs")
		os.makedirs(log_directory, exist_ok=True)
		name = args.name
		if name == "":
			name = "_"
		logfile_path = os.path.join(log_directory, name + ".log")
		with open(logfile_path, "a") as logfile:
			logfile.write(f"=== training started at {datetime.datetime.now()} ===\n")
			with tqdm(desc="Training", total=n_steps, unit="step") as t:
				while testbed.frame():
					if testbed.want_repl():
						repl(testbed)

					trained_steps = testbed.training_step

					if testbed.training_step % 100 == 0:
						logfile.write(f"{time.time()} {testbed.training_step} {testbed.loss}\n")

					if testbed.shall_train is True and args.save_snapshot and (t.format_dict['elapsed'] + time_offset) >= timepoints[current_timepoint]:
						snapshot_save_time += save_snapshot(str(timepoints[current_timepoint]) + "s")
						current_timepoint = current_timepoint + 1
						if current_timepoint >= timepoint_count:
							break

					if args.save_per_n > 0 and trained_steps > 0 and trained_steps % args.save_per_n == 0:
						snapshot_save_time += save_snapshot(trained_steps)

					if args.max_time and t.format_dict['elapsed'] + time_offset >= args.max_time:
						break

					# What will happen when training is done?
					if testbed.training_step >= n_steps:
						if args.gui:
							testbed.shall_train = False
						else:
							break

					# Update progress bar
					if testbed.training_step < old_training_step or old_training_step == 0:
						old_training_step = 0
						t.reset()

					now = time.monotonic()
					if now - tqdm_last_update > 0.1:
						t.update(testbed.training_step - old_training_step)
						t.set_postfix(loss=testbed.loss)
						old_training_step = testbed.training_step
						tqdm_last_update = now

	print(f"Save snapshot consumed: {snapshot_save_time}")

	if args.save_snapshot and trained_steps > 0:
		print("Saving snapshot ", args.save_snapshot)
		testbed.save_snapshot(os.path.join(args.save_snapshot, str(trained_steps) + ".msgpack"), False)

	if args.test_transforms:
		if os.path.exists(args.test_transforms) == False:
			args.test_transforms = os.path.join(os.path.dirname(args.scene), args.test_transforms)

		save_dir = os.path.join(os.path.dirname(args.scene), "test_transforms", args.name)
		os.makedirs(save_dir, exist_ok=True)

		print("Evaluating test transforms from ", args.test_transforms)
		with open(args.test_transforms) as f:
			test_transforms = json.load(f)
		data_dir=os.path.dirname(args.test_transforms)
		totmse = 0
		totpsnr = 0
		totssim = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on black background
		testbed.background_color = [0.0, 0.0, 0.0, 1.0]

		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.rendering_min_transmittance = 1e-4

		testbed.fov_axis = 0
		testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
		testbed.shall_train = False

		args.test_max = int(args.test_max)
		total_frame_count = len(test_transforms["frames"])
		if args.test_max == 0:
			args.test_max = total_frame_count
		test_per_n = total_frame_count // args.test_max

		with open(os.path.join(save_dir, "PSNR-SSIM.txt"), "w") as f:
			with tqdm(list(range(0, args.test_max)), unit="images", desc=f"Rendering test frame") as t:
				for i in t:
					frame = test_transforms["frames"][test_per_n * i]

					p = frame["file_path"]
					if "." not in p:
						p = p + ".png"
					ref_fname = os.path.join(data_dir, p)
					if not os.path.isfile(ref_fname):
						ref_fname = os.path.join(data_dir, p + ".png")
						if not os.path.isfile(ref_fname):
							ref_fname = os.path.join(data_dir, p + ".jpg")
							if not os.path.isfile(ref_fname):
								ref_fname = os.path.join(data_dir, p + ".jpeg")
								if not os.path.isfile(ref_fname):
									ref_fname = os.path.join(data_dir, p + ".exr")

					ref_image = read_image(ref_fname)

					# NeRF blends with background colors in sRGB space, rather than first
					# transforming to linear space, blending there, and then converting back.
					# (See e.g. the PNG spec for more information on how the `alpha` channel
					# is always a linear quantity.)
					# The following lines of code reproduce NeRF's behavior (if enabled in
					# testbed) in order to make the numbers comparable.
					if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
						# Since sRGB conversion is non-linear, alpha must be factored out of it
						ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
						ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
						ref_image[...,:3] *= ref_image[...,3:4]
						ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
						ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

					filename = os.path.basename(frame["file_path"])

					testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
					image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)
					write_image(os.path.join(save_dir, "{}_out.png".format(filename)), image)

					if ref_image.shape[2] == 3:
						image = image[:, :, :-1]

					diffimg = np.absolute(image - ref_image)
					diffimg[...,3:4] = 1.0

					write_image(os.path.join(save_dir, "{}_ref.png".format(filename)), ref_image)
					write_image(os.path.join(save_dir, "{}_diff.png".format(filename)), diffimg)

					A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
					R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
					mse = float(compute_error("MSE", A, R))
					ssim = float(compute_error("SSIM", A, R))
					totssim += ssim
					totmse += mse
					psnr = mse2psnr(mse)
					totpsnr += psnr
					minpsnr = psnr if psnr<minpsnr else minpsnr
					maxpsnr = psnr if psnr>maxpsnr else maxpsnr
					totcount = totcount+1
					t.set_postfix(psnr = totpsnr/(totcount or 1))

					f.write(f"{filename}: PSNR={psnr:.3f}, SSIM={ssim:.3f}\n")

			psnr_avgmse = mse2psnr(totmse/(totcount or 1))
			psnr = totpsnr/(totcount or 1)
			ssim = totssim/(totcount or 1)
			print(f"PSNR={psnr:.3f} [min={minpsnr:.3f} max={maxpsnr:.3f}] SSIM={ssim:.3f}")
			f.write(f"PSNR={psnr:.3f} [min={minpsnr:.3f} max={maxpsnr:.3f}] SSIM={ssim:.3f}")

	if args.save_mesh:
		res = args.marching_cubes_res or 256
		print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res])

	if ref_transforms:
		testbed.fov_axis = 0
		testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
		if not args.screenshot_frames:
			args.screenshot_frames = range(len(ref_transforms["frames"]))
		print(args.screenshot_frames)
		for idx in args.screenshot_frames:
			f = ref_transforms["frames"][int(idx)]
			cam_matrix = f["transform_matrix"]
			testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
			outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

			# Some NeRF datasets lack the .png suffix in the dataset metadata
			if not os.path.splitext(outname)[1]:
				outname = outname + ".png"

			print(f"rendering {outname}")
			image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
			os.makedirs(os.path.dirname(outname), exist_ok=True)
			write_image(outname, image)
	elif args.screenshot_dir:
		outname = os.path.join(args.screenshot_dir, args.scene + "_" + network_stem)
		print(f"Rendering {outname}.png")
		image = testbed.render(args.width or 1920, args.height or 1080, args.screenshot_spp, True)
		if os.path.dirname(outname) != "":
			os.makedirs(os.path.dirname(outname), exist_ok=True)
		write_image(outname + ".png", image)

	if args.video_camera_path:
		testbed.load_camera_path(args.video_camera_path)

		resolution = [args.width or 1920, args.height or 1080]
		n_frames = args.video_n_seconds * args.video_fps

		if "tmp" in os.listdir():
			shutil.rmtree("tmp")
		os.makedirs("tmp")

		for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
			testbed.camera_smoothing = args.video_camera_smoothing and i > 0
			frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
			write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

		os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")
		shutil.rmtree("tmp")
