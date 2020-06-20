#!/usr/bin/python

# importing libraries
import numpy as np
import cv2
import time
import sys
import matplotlib.pyplot as plt

# adopted from imutils library (https://github.com/jrosebr1/imutils/)
def grab_contours(cnts): 
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

# this function returns a cropped version of the stiched image which doesnot contain black borders
# Note: I have not considered all the edge cases here
def improve_stiching_result(im):
	img = im.copy()
	im[np.where((im==[0,0,0]).all(axis=2))] = [255,255,255];
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 250, 255, 0)
	im2,contours,heirarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) != 0:
	    # find the biggest countour (c) by the area
	    c = max(contours, key = cv2.contourArea)
	    x,y,w,h = cv2.boundingRect(c)

	return img[0:y+h,0:x,:]    

def main():
	img_name_list = []
	for i in range(1,len(sys.argv)):
		img_name_list.append(sys.argv[i])

	img1_orig = cv2.imread(img_name_list[1])
	img2_orig = cv2.imread(img_name_list[0])  

	for j in range(2,len(sys.argv)):
		begin = time.clock()

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

			plt.subplot(1,3,1)
			plt.imshow(cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB))

			plt.subplot(1,3,2)
			plt.imshow(cv2.cvtColor(img2_orig, cv2.COLOR_BGR2RGB))

			plt.subplot(1,3,3)
			plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

			plt.show()
			

		# # improving the stcihing result
		result = improve_stiching_result(result)

		cv2.imwrite('results/'+str(j-1)+'_stiched.jpg',result)

		# updating img_orgs
		if j<len(img_name_list):
			img1_orig = cv2.imread(img_name_list[j])
			img2_orig = result

if __name__ == '__main__':
	main()			



   
                          