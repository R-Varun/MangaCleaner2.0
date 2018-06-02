import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import sys
import json
import os

INPUT_DIR = "./Region/raw"
OUTPUT_DIR = "./Region/bbox"
xdata = np.linspace(0, 9 * np.pi, num=301)
ydata = np.sin(xdata) * np.cos(xdata * 2.4)

fig, ax = plt.subplots()
# line, = ax.plot(xdata, ydata)
# point, = ax.plot([], [], marker="o", color="crimson")
# text = ax.text(0, 0, "")

global curInd
global dirList
curInd = 0


dirList = os.listdir(INPUT_DIR)
break_off = 0
index = 0
newDir = []
for i in dirList:
	name = i[:i.index(".")]
	matches = list(filter(lambda x: x.startswith(name), os.listdir(OUTPUT_DIR)))
	if len(matches) == 0:
		newDir.append(i)

dirList = newDir



plt.imshow(plt.imread(os.path.join(INPUT_DIR, dirList[curInd])))

global cur_region
global regions
regions = set([])
global patches
patches = []

def press(event):
	global cur_region
	global regions
	global curInd
	global dirList
	global patches
	print('press', event.key)
	sys.stdout.flush()
	if event.key == 'x':
		print("saved")
		if cur_region != None:
			regions.add(cur_region)
			rect = plt.Rectangle((min(cur_region[0], cur_region[1]), min(cur_region[2], cur_region[3])), np.abs(cur_region[1] - cur_region[0]), np.abs(cur_region[2] - cur_region[3]))
			patches.append(rect)
			ax.add_patch(rect)
			print(regions)

	if event.key == 'escape':


		name = dirList[curInd][:dirList[curInd].index(".")]
		with open(os.path.join(OUTPUT_DIR, name+ ".json"), 'w') as outfile:
			json.dump(list(regions), outfile)
		for i in patches:
			i.remove()
		patches = []

		regions = set([])
		curInd += 1

		plt.imshow(plt.imread(os.path.join(INPUT_DIR, dirList[curInd])))





fig.canvas.mpl_connect('key_press_event', press)

def line_select_callback(eclick, erelease):
	x1, y1 = eclick.xdata, eclick.ydata
	x2, y2 = erelease.xdata, erelease.ydata
	global cur_region
	cur_region = (x1, x2, y1, y2)




rs = RectangleSelector(ax, line_select_callback,
					   drawtype='box', useblit=False, button=[1],
					   minspanx=5, minspany=5, spancoords='pixels',
					   interactive=True)

plt.show()
