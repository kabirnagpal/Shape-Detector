#works only for shapes on black background
# import the necessary packages
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from pyimagesearch.shapedetector import ShapeDetector


# load the image
image = plt.imread('s1.jpg')      #always add complete path
cv2.imshow("Actual Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


cv2.imshow("shapemask", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	if M["m00"] != 0:
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
	else:
		cX, cY = 0, 0 # set values as what you need in the situation
	shape = sd.detect(c)
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
    
cv2.destroyAllWindows()
