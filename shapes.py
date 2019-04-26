#works only for shapes on black background
# import the necessary packages
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt


# load the image
image = plt.imread('/home/jak/Desktop/open-cv/projects/shapedetection/geometric-shapes-polygons.jpg')      #always add complete path
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.imshow("Actual Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

lower = np.array([0, 0, 0])
upper = np.array([220, 220, 220])
shapeMask = cv2.inRange(image, lower, upper)
image=shapeMask
shapeMask=cv2.bitwise_not(shapeMask)

cv2.imshow("modified original", image)
#cv2.imshow("shapemask", shapeMask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# find the contours in the mask
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)
cv2.waitKey(0)
cv2.destroyAllWindows()


# loop over the contours
for c in cnts:
	# draw the contour and show it
    cv2.drawContours(image, [c], -1, (200, 200, 200), 5)

    #cv2.imshow("Image", image)
    #if cv2.waitKey(0)==27:
    #    break

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
