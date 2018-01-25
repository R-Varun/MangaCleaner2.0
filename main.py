import cv2
import numpy as np
import classify
from classify import resize
from convnet import predict
from PIL import Image


# classifier = classify.create_classifier2()
def getClassifierPrediction(image):
    return predict(image)

image = cv2.imread("./ONSIES/raw2.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale

# ret,gray = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
cv2.imshow('image',gray)
cv2.waitKey(0)



gray = cv2.GaussianBlur(gray, (5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,111,50) # threshold

ret,thresh = cv2.threshold(thresh,230,255,cv2.THRESH_BINARY_INV)

cv2.imshow('image',thresh)
cv2.waitKey(0)

thresh = cv2.Canny(thresh, 500, 700)



thresh = cv2.medianBlur(thresh, 3)

#hopefully this would get rid of some noise as text is relatively dense

cv2.imshow('image',thresh)
cv2.waitKey(0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE ,(3,7))
# dilated = cv2.dilate(thresh,kernel,iterations = 5  ) # dilate
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 4)
# thresh = cv2.erode(thresh,kernel,iterations = 2)
# cv2.imshow('image',thresh)
# cv2.waitKey(0)
# dilated = thresh
dilated = cv2.dilate(thresh,kernel,iterations = 3)
cv2.imshow('image',dilated)
cv2.waitKey(0)
s, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
# RETR_EXTERNAL
# for each contour found, draw a rectangle around it on original image
useC = 1
if useC == 1:
    dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        # if h>500 and w>500:
        #     continue

        pertinent = image[y:y+h, x:x+w]
        pil_im = Image.fromarray(pertinent)
        isValid = getClassifierPrediction(pil_im)
        print(isValid)

        if isValid == 1:
            print("this one is valid")
            image[y:y + h, x:x + w] = [255,255,255]
            # cv2.rectangle(image, (x, y), (x + w,  y + h), (0, 255, 0), 2)
            # cv2.rectangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # print("weeded out")
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # cv2.rectangle(dilated, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.rectangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)


if not useC:
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10;    # the graylevel of images
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
    dilated = cv2.drawKeypoints(dilated, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('image',dilated)
    cv2.waitKey(0)


elif useC == 2:

    r, c, ch  = image.shape
    for x in range(c - 20):
        for y in range(r - 40):
            w = 10
            h = 30
            pertinent = image[y:y + 30, x:x + 10]
            pil_im = Image.fromarray(pertinent)
            isValid = getClassifierPrediction(pil_im)
            # print(isValid)

            if isValid == 1:
                print("this one is valid")
                # image[y:y + h, x:x + w] = [255, 255, 255]
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                print("weeded out")

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.re/ctangle(dilated, (x, y), (x + w, y + h), (0, 255, 0), 2)


# write original image with added contours to disk
cv2.imwrite("contoured.jpg", image)
cv2.imwrite("dilated.jpg", dilated)


