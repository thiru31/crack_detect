
import numpy as np
import cv2

from matplotlib import pyplot as plt
import numpy

  img = cv2.imread(r'(PATH)')
#img = cv2.imread(r'C:\Users\Thirumalai N\Downloads\crack-detection-opencv-master\crack-detection-opencv-master\test\test (6).JPG')



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray,(3,3))


numpy.seterr(divide = 'ignore')
img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255
img_log = np.array(img_log,dtype=np.uint8)
np.seterr(invalid='ignore')

bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)
edges = cv2.Canny(bilateral,10,50)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
orb = cv2.ORB_create(nfeatures=1500)
keypoints, descriptors = orb.detectAndCompute(closing, None)
featuredImg = cv2.drawKeypoints(closing, keypoints, None)

#OUTPUT image
cv2.imwrite(r'C:\Users\Thirumalai N\Downloads\crack-detection-opencv-master\crack-detection-opencv-master\test\out.jpg', featuredImg)


#PLOTTING POINTS ON CRACK

plt.subplot(121),plt.imshow(img)
plt.title('Original'),plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(featuredImg,cmap='gray')
plt.title('Output Image'),plt.xticks([]), plt.yticks([])
plt.show()
