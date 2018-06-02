import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import sys
import json
import os
import utils
from PIL import Image

INPUT_DIR = "./Region/raw"
OUTPUT_DIR = "./Region/bbox"

SAVE_POS_DIR = "./temp/pos"


SAVE_NEG_DIR = "./temp/neg"



CREATE_NEGATIVE_IMAGES = False


fig, ax = plt.subplots()
# line, = ax.plot(xdata, ydata)
# point, = ax.plot([], [], marker="o", color="crimson")
# text = ax.text(0, 0, "")

completed_annotations = os.listdir(OUTPUT_DIR)
available_images = os.listdir(INPUT_DIR)

for item in completed_annotations:

	name = item[:item.index(".")]
	ext = item[item.index(".") + 1:]


	wantedImage = list(filter(lambda x: x.startswith(name), available_images))[0]
	wantedImage = np.asarray(Image.open(os.path.join(INPUT_DIR, wantedImage)))
	# if ext == "jpg":
	# 	b, g, r = wantedImage.split()
	# 	wantedImage = Image.merge("RGB", (r, g, b))
	# wantedImage = wantedImage * 255
	bboxes = json.loads(open(os.path.join(OUTPUT_DIR, item), "r").read())

	# create array of rects to check for intersection with sliding windows
	rectArr = []

	for i, cur_region in enumerate(bboxes):
		# rect = plt.Rectangle((min(cur_region[0], cur_region[1]), min(cur_region[2], cur_region[3])),
		# 					 np.abs(cur_region[1] - cur_region[0]), np.abs(cur_region[2] - cur_region[3]))
		#
		#
		cur_region = list(map(lambda x : int(x), cur_region))

		xmin = min(cur_region[0], cur_region[1])
		xmax = max(cur_region[0], cur_region[1])
		ymin = min(cur_region[2], cur_region[3])
		ymax = max(cur_region[2], cur_region[3])

		result = Image.fromarray((wantedImage[ymin:ymax, xmin:xmax]).astype(np.uint8))
		result.save(os.path.join(SAVE_POS_DIR, name +"-" + str(i) + ".png"))

	# 	save rectangle
		rectArr.append(utils.Rectangle(xmin, ymin, xmax, ymax))




	neg_count = 0
	if CREATE_NEGATIVE_IMAGES:
		width = wantedImage.shape[1] // 15
		height = wantedImage.shape[0] // 15
		for window in utils.sliding_window(wantedImage, (width // 2, height // 2), windowSize=(width, height)):
			window_rect = utils.Rectangle(window[0], window[1], window[0] + width, window[1] + height)
			valid = True
			for i in rectArr:
				if window_rect.intersects(i):
					valid = False
					break

			if valid == False:
				continue

			result = Image.fromarray((window[2]).astype(np.uint8))

			result.save(os.path.join(SAVE_NEG_DIR, name + "-" + str(neg_count) + ".png"))
			neg_count += 1



		# ax.add_patch(rect)
	# plt.imshow(wantedImage)
	# plt.show()


