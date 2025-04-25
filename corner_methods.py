import cv2
import numpy as np

def harris(image, args=None):

    args = args or {'blockSize': 2, 'ksize': 3, 'k': 0.04}
    corners = cv2.cornerHarris(image, **args)
    corners = cv2.dilate(corners, None)
    corners = np.column_stack(np.where(corners > 0.01 * corners.max()))
    return corners[:, ::-1].astype(np.float32)

def shi_tomasi(image, args=None):

    args = args or {'maxCorners': 25, 'qualityLevel': 0.01, 'minDistance': 10}
    corners = cv2.goodFeaturesToTrack(image, **args)
    return corners[:, 0].astype(np.float32) if corners is not None else np.array([])

def fast(image, args=None):

    args = args or {'threshold': 25, 'type': cv2.FAST_FEATURE_DETECTOR_TYPE_7_12}
    fast = cv2.FastFeatureDetector_create(**args)
    keypoints = fast.detect(image, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)

def orb(image, args=None):

    args = args or {
        'nfeatures': 500, 'scaleFactor': 1.1, 'nlevels': 10,
        'edgeThreshold': 31, 'firstLevel': 0, 'WTA_K': 2,
        'scoreType': cv2.ORB_HARRIS_SCORE, 'patchSize': 31, 'fastThreshold': 20
    }
    orb = cv2.ORB_create(**args)
    keypoints = orb.detect(image, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)

def sift(image, args=None):

    args = args or {
        'nOctaveLayers': 3, 'contrastThreshold': 0.04, 'edgeThreshold': 10, 'sigma': 1.6
    }
    sift = cv2.SIFT_create(**args)
    keypoints, _ = sift.detectAndCompute(image, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)

def brisk(image, args=None):

    args = args or {'thresh': 30, 'octaves': 3, 'patternScale': 1.0}
    brisk = cv2.BRISK_create(**args)
    keypoints, _ = brisk.detectAndCompute(image, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)

def agast(image, args=None):

    args = args or {'threshold': 10, 'type': 1}
    agast = cv2.AgastFeatureDetector_create(**args)
    keypoints = agast.detect(image, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)

def kaze(image, args=None):

    args = args or {'threshold': 0.001, 'nOctaves': 3, 'nOctaveLayers': 3, 'diffusivity': 1}
    kaze = cv2.KAZE_create(**args)
    keypoints, _ = kaze.detectAndCompute(image, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)

def akaze(image, args=None):

    args = args or {'threshold': 0.001, 'nOctaves': 4, 'nOctaveLayers': 4, 'diffusivity': 2}
    akaze = cv2.AKAZE_create(**args)
    keypoints, _ = akaze.detectAndCompute(image, None)
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)