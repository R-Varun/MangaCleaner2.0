import string
import random
import os
import cv2
dir_path = os.path.dirname(os.path.realpath(__file__))


neg = os.path.join(dir_path, "test")


def rnd(N=10):
	return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))


def sliceImg(img, output, slices=20):
	height, width, channels = img.shape

	incW = int(width / slices)
	incH = int(height / slices)

	counter = 0
	for i in range(slices):
		for j in range(slices):
			y = j * incH
			x = i * incW

			slice = img[y:y + incH, x:x + incW]
			counter += 1
			# cv2.imshow("l", slice)
			# cv2.waitKey(0)
			cv2.imwrite(os.path.join(output, rnd() + ".png"), slice)

	print(counter)

sliceImg(cv2.imread("./chopthis.jpg"), os.path.join(neg, "neg"))




