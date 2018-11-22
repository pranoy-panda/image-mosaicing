import numpy as np
import cv2
import time

begin = time.clock()

img1_orig = cv2.imread('images/ec2.jpeg')
img2_orig = cv2.imread('images/ec3.jpeg')  
img1 = cv2.cvtColor(img1_orig,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_orig,cv2.COLOR_BGR2GRAY)

#creating a SIFT object
sift = cv2.xfeatures2d.SIFT_create()

#finding keypoints and descriptors with SIFT
(kp1,des1) = sift.detectAndCompute(img1,None)
(kp2,des2) = sift.detectAndCompute(img2,None)

# creating BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2) # brute force search for each descriptor in img1 to find the most 
                                #similar descriptor in img2
matches = bf.knnMatch(des1,des2, k=2) # 2 nearest matches for each descriptor

'''
Keypoints between two images are matched by identifying their nearest neighbours. 
But in some cases, the second closest-match may be very near to the first. It may 
happen due to noise or some other reasons. In that case, ratio of closest-distance 
to second-closest distance is taken. If it is greater than 0.8, they are rejected.
It eliminaters around 90percent of false matches while discards only 5 percent 
correct matches, as per the paper.
'''
# Apply ratio test
good_matches = []
good = []
for m,n in matches:
	#m is the closest dist. and n the 2nd closest
    if (m.distance/n.distance) < 0.8:
        good_matches.append((m.trainIdx,m.queryIdx))
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
#img3 = cv2.drawMatchesKnn(img1_orig,kp1,img2_orig,kp2,good,None,flags=2)  

kp1 = np.float32([kp.pt for kp in kp1])   
kp2 = np.float32([kp.pt for kp in kp2])

reprojThresh = 4.0
# computing a homography requires at least 4 matches
if len(matches) > 4:
	# construct the two sets of points
	pts1 = np.float32([kp1[i] for (_, i) in good_matches])
	pts2 = np.float32([kp2[i] for (i, _) in good_matches])

	# computing the homography between the two sets of points
	(H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC,reprojThresh)

	result = cv2.warpPerspective(img1_orig, H,(img1_orig.shape[1] + img2_orig.shape[1], img1_orig.shape[0]))
	result[0:img2_orig.shape[0], 0:img2_orig.shape[1]] = img2_orig  

	print 'time for execution: '+str(time.clock()-begin)
	cv2.imshow('result',result)
	cv2.waitKey(0)
	#cv2.imwrite('room2.jpg',result) 
	#cv2.imwrite('matching.jpg',img3)   
   
                          