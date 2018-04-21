import cv2
import numpy as np
import classify
from classify import resize
from convnet import predict
from PIL import Image
import matplotlib.pyplot as plt

# classifier = classify.create_classifier2()
def getClassifierPrediction(image):
	return predict(image)


useC = 1
image = cv2.imread("./ONSIES/raw2.jpg")

if useC == 1:


	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale

	# ret,gray = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
	cv2.imshow('image', gray)
	cv2.waitKey(0)

	# gray = cv2.GaussianBlur(gray, (5, 5),  0)
	# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 111, 50)  # threshold
	thresh = gray

	thresh = cv2.Canny(thresh, 400, 1000)
	# thresh = cv2.Laplacian(thresh,cv2.CV_8U )
	thresh = cv2.medianBlur(thresh, 3)
	cv2.imshow('huh', thresh)
	cv2.waitKey(0)

	# hopefully this would get rid of some noise as text is relatively dense

	cv2.imshow('image', thresh)
	cv2.waitKey(0)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

	dilated = thresh
	dilated = cv2.dilate(dilated, kernel, iterations=3)
	cv2.imshow('image', dilated)
	cv2.waitKey(0)

	s, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )  # get contours
	dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
	for contour in contours:
		[x, y, w, h] = cv2.boundingRect(contour)
		# if h>500 and w>500:
		#     continue

		pertinent = image[y:y + h, x:x + w]
		pil_im = Image.fromarray(pertinent)
		isValid = getClassifierPrediction(pil_im)
		print(isValid)

		if isValid == 1:
			# print("this one is valid")
			# image[y:y + h, x:x + w] = [255, 255, 255]
			cv2.rectangle(image, (x, y), (x + w,  y + h), (0, 255, 0), 2)
			cv2.rectangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)
		else:
			# pass
			# print("weeded out")
	# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# cv2.rectangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
	# cv2.rectangle(dilated, (x, y), (x + w, y + h), (255, 0, 255), 2)

			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
			cv2.rectangle(dilated, (x, y), (x + w, y + h), (255, 0, 255), 2)
if not useC:
	params = cv2.SimpleBlobDetector_Params()
	# Change thresholds
	params.minThreshold = 10;  # the graylevel of images
	params.maxThreshold = 200;
	# Disable unwanted filter criteria params
	params.filterByInertia = False
	params.filterByConvexity = False
	#
	params.filterByCircularity = False
	# params.minCircularity = 0.1

	params.filterByColor = True
	params.blobColor = 255

	# Filter by Area
	params.filterByArea = True
	params.minArea = 1000

	detector = cv2.SimpleBlobDetector_create(params)
	keypoints = detector.detect(dilated)
	dilated = cv2.drawKeypoints(dilated, keypoints, np.array([]), (0, 0, 255),
								cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow('image', dilated)
	cv2.waitKey(0)


elif useC == 2:

	m_w, m_h = 50, 50

	r, c, ch = image.shape
	trans_x, trans_y = 300, 500
	ni = cv2.resize(image, (trans_x, trans_y))


	for x in range(0,trans_x - m_w, 40):
		for y in range(0,trans_y - m_h, 40):
			for w in range(1,m_w, 40):
				for h in range(1, m_h, 40):

					pertinent = ni[y:y + h, x:x + w]
					pil_im = Image.fromarray(pertinent)
					isValid = getClassifierPrediction(pil_im)
					# print(isValid)
					if isValid == 1:
						# print("this one is valid")
						# image[y:y + h, x:x + w] = [255, 255, 255]
						t_x = int(c / trans_x ) * x
						t_y = int(r / trans_y ) * y

						t_w = int(c / trans_x) * w
						t_h = int(r / trans_y) * h
						cv2.rectangle(image, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 0), 2)
						cv2.rectangle(ni, (x, y), (x + w, y + h), (0, 255, 0), 2)

					# cv2.rectangle(dilated, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 255, 0), 2)
					else:
						pass
						# print("weeded out")

						# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
						# cv2.re/ctangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.imwrite("contoured-min.jpg", ni)

# write original image with added contours to disk
cv2.imwrite("contoured.jpg", image)
cv2.imwrite("dilated.jpg", dilated)
