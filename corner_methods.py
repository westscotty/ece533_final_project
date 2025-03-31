import cv2
import numpy as np
import os


def harris(image, args=[], corners=[]):
    corners = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
    corners = cv2.dilate(corners, None)
    corners = np.column_stack(np.where(corners > 0.01 * corners.max()))
    return corners[:, ::-1]

def shi_tomasi(image, args=[], corners=[]):
    corners = cv2.goodFeaturesToTrack(image, maxCorners=25, qualityLevel=0.01, minDistance=10, blockSize=2, k=0.04)
    return corners[:,0]

def fast(image, args=[], corners=[]):
    types = [cv2.FAST_FEATURE_DETECTOR_TYPE_5_8, cv2.FAST_FEATURE_DETECTOR_TYPE_7_12, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16]
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True, type=types[1])
    keypoints = fast.detect(image, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return corners

def orb(image, args=[], corners=[]):
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.1, nlevels=10, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    keypoints = orb.detect(image, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return corners

def sift(image, args=[], corners=[]):
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return corners #, descriptors

def brisk(image, args=[], corners=[]):
    brisk = cv2.BRISK_create(thresh=30, octaves=3, patternScale=1.0)
    keypoints, descriptors = brisk.detectAndCompute(image, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return corners #, descriptors

def agast(image, args=[], corners=[]):
    # types = [cv2.AgastFeatureDetector_TYPE_5_8, cv2.AgastFeatureDetector_TYPE_7_12, cv2.AgastFeatureDetector_TYPE_9_16]
    agast = cv2.AgastFeatureDetector_create(threshold=10, nonmaxSuppression=True, type=1)
    keypoints = agast.detect(image, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return corners

def kaze(image, args=[], corners=[]):
    kaze = cv2.KAZE_create(threshold=0.001, nOctaves=3, nOctaveLayers=3, diffusivity=1)
    keypoints, descriptors = kaze.detectAndCompute(image, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)    
    return corners #, descriptors

def akaze(image, args=[], corners=[]):
    akaze = cv2.AKAZE_create(threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=2)
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return corners #, descriptors

# ## Can't install it because of patent issue ...
# def surf(image, args=[], corners=[]):
#     surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, nOctaves=4, nOctaveLayers=4, extended=False, upright=False)
#     keypoints, descriptors = surf.detectAndCompute(image, None)
#     corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
#     return corners #, descriptors
