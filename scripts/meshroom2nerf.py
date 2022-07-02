import argparse
import json
import os.path

import numpy as np
import math


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--sfmfile", required=True, help="path to cameras.sfm file")
	parser.add_argument("--outfile", required=True, help="path to save NeRF transforms json file")
	parser.add_argument("--imgpath", required=True, help="images path relative to transforms json file")
	args = parser.parse_args()
	return args


def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa, da, ob,
						  db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c) ** 2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa + ta * da + ob + tb * db) * 0.5, denom


args = parse_args()

with open(args.sfmfile, "r") as sfmf:
	sfm = json.load(sfmf)

poseMap = dict()

for pose in sfm["poses"]:
	poseMap[pose["poseId"]] = pose

frames = []

up = np.zeros(3)
for view in sfm["views"]:
	viewId = view["viewId"]
	poseId = view["poseId"]
	path = view["path"]
	try:
		pose = poseMap[poseId]["pose"]
	except:
		continue
	rotation = pose["transform"]["rotation"]
	center = pose["transform"]["center"]

	rotation = np.array([
		[
			float(rotation[0]),
			float(rotation[3]),
			float(rotation[6]),
			0.0,
		],
		[
			float(rotation[1]),
			float(rotation[4]),
			float(rotation[7]),
			0.0,
		],
		[
			float(rotation[2]),
			float(rotation[5]),
			float(rotation[8]),
			0.0,
		],
		[
			0.0,
			0.0,
			0.0,
			1.0,
		]
	])

	c = np.array([
		[
			1.0,
			0.0,
			0.0,
			-float(center[0]),
		],
		[
			0.0,
			1.0,
			0.0,
			-float(center[1]),
		],
		[
			0.0,
			0.0,
			1.0,
			-float(center[2]),
		],
		[
			0.0,
			0.0,
			0.0,
			1.0,
		],
	])

	extrinsic = np.matmul(rotation, c)

	c2w = np.linalg.inv(extrinsic)
	c2w[0:3, 2] *= -1  # flip the y and z axis
	c2w[0:3, 1] *= -1
	c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
	c2w[2, :] *= -1  # flip whole world upside down

	up += c2w[0:3, 1]

	frames.append({
		"file_path": os.path.join(args.imgpath, os.path.basename(path)),
		"transform_matrix": c2w,
	})

nframes = len(frames)

up = up / np.linalg.norm(up)
print("up vector was", up)
R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
R = np.pad(R, [0, 1])
R[-1, -1] = 1

for f in frames:
	f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

# find a central point they are all looking at
print("computing center of attention...")
totw = 0.0
totp = np.array([0.0, 0.0, 0.0])
for f in frames:
	mf = f["transform_matrix"][0:3, :]
	for g in frames:
		mg = g["transform_matrix"][0:3, :]
		p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
		if w > 0.01:
			totp += p * w
			totw += w
totp /= totw
print(totp)  # the cameras are looking at totp
for f in frames:
	f["transform_matrix"][0:3, 3] -= totp

avglen = 0.
for f in frames:
	avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
avglen /= nframes
print("avg camera distance from origin", avglen)
for f in frames:
	f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

for f in frames:
	f["transform_matrix"] = f["transform_matrix"].tolist()

intrinsics = sfm["intrinsics"][0]

# w and h
w = float(intrinsics["width"])
h = float(intrinsics["height"])

# fl_x and fl_y
fl_x = float(intrinsics["pxFocalLength"])
fl_y = fl_x

# cx and cy
cx = float(intrinsics["principalPoint"][0])
cy = float(intrinsics["principalPoint"][1])

angle_x = math.atan(w / (fl_x * 2)) * 2
angle_y = math.atan(h / (fl_y * 2)) * 2
fovx = angle_x * 180 / math.pi
fovy = angle_y * 180 / math.pi

out = {
	"camera_angle_x": angle_x,
	"camera_angle_y": angle_y,
	"fl_x": fl_x,
	"fl_y": fl_y,
	"cx": cx,
	"cy": cy,
	"w": w,
	"h": h,
	"aabb_scale": 16,
	"frames": frames,
}

with open(args.outfile, "w") as f:
	json.dump(out, f, indent=4)
