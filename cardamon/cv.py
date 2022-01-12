import cv2
from google.colab.patches import cv2_imshow
src = cv2.imread("/content/green-cardamom-pods-isolated-on-600w-375763168.webp")

src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY);
src_gray = cv2.blur(src_gray, (3,3));

cv2_imshow(src_gray)

edged = cv2.Canny(src_gray, 30, 200)
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(src, contours, -1, (0, 255, 0), 3)
cv2_imshow(src)

canny_output = cv2.Canny(src_gray, 100, 100 * 2)

import numpy as np
import random as rng
rng.seed(12345)

minRect = [None]*len(contours)
minEllipse = [None]*len(contours)
for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv2.fitEllipse(c)
    # Draw contours + rotated rects + ellipses
    
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

for i, c in enumerate(contours):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    # contour
    cv2.drawContours(drawing, contours, i, color)
    # ellipse
    if c.shape[0] > 5:
        cv2.ellipse(drawing, minEllipse[i], color, 2)
    # rotated rectangle
    box = cv2.boxPoints(minRect[i])
    box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    cv2.drawContours(drawing, [box], 0, color)


cv2_imshow( drawing)

