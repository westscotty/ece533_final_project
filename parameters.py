from corner_methods import *
import os

image_path = os.path.join(os.getcwd(), "data/Urban_Corner_datasets/Images")
ground_truth_path = os.path.join(os.getcwd(), "data/Urban_Corner_datasets/Ground_Truth")
output_path = os.path.join(os.getcwd(), "results")

SCALES = [0.5, 1.0, 2.0, 3.0]  # Scales to test

ALGORITHMS = {
    'Harris': harris,
    'Shi-Tomasi': shi_tomasi,
    'FAST': fast,
    'ORB': orb,
    'SIFT': sift,
    'BRISK': brisk,
    'AGAST': agast,
    'KAZE': kaze,
    'AKAZE': akaze
}

PARAM_GRIDS = {
    'Harris': {
        'blockSize': [2, 3, 5, 7, 9, 11],  # Neighborhood size for corner detection
        'ksize': [3, 5, 7, 9, 11],         # Sobel kernel size for gradient computation
        'k': [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15],  # Harris detector free parameter
        'borderType': [cv2.BORDER_DEFAULT, cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT]  # Border handling
    },
    'Shi-Tomasi': {
        'maxCorners': [25, 50, 100, 200, 500, 1000],  # Maximum number of corners to detect
        'qualityLevel': [0.005, 0.01, 0.05, 0.1, 0.2],  # Minimum quality threshold
        'minDistance': [3, 5, 10, 15, 20, 25, 30],   # Minimum distance between corners
        'blockSize': [3, 5, 7, 9]                    # Neighborhood size for eigenvalue computation
    },
    'FAST': {
        'threshold': [5, 10, 20, 25, 50, 75, 100],   # Intensity difference threshold
        'type': [
            cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
            cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
            cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        ],  # FAST detector variants
        'nonmaxSuppression': [True, False]            # Enable/disable non-maximum suppression
    },
    'ORB': {
        'nfeatures': [100, 500, 1000, 2000, 5000],   # Maximum number of features
        'scaleFactor': [1.05, 1.1, 1.2, 1.3, 1.5],   # Pyramid scale factor
        'nlevels': [4, 8, 10, 12, 16],              # Number of pyramid levels
        'edgeThreshold': [15, 31, 50, 70],           # Border size where features are not detected
        'patchSize': [15, 31, 50],                   # Size of patch for descriptor
        'fastThreshold': [10, 20, 30, 50]            # FAST threshold for keypoint detection
    },
    'SIFT': {
        'nOctaveLayers': [2, 3, 4, 6],              # Layers per octave
        'contrastThreshold': [0.02, 0.04, 0.08, 0.16],  # Contrast threshold for filtering
        'edgeThreshold': [5, 10, 20, 30],           # Edge threshold for filtering
        'sigma': [1.2, 1.6, 2.0, 2.4]               # Gaussian blur sigma for initial image
    },
    'BRISK': {
        'thresh': [10, 30, 50, 70, 100],            # AGAST detection threshold
        'octaves': [2, 3, 4, 6],                    # Number of octaves
        'patternScale': [0.5, 1.0, 1.5, 2.0]        # Scale applied to the pattern
    },
    'AGAST': {
        'threshold': [5, 10, 20, 30, 50],           # Intensity difference threshold
        'type': [
            cv2.AgastFeatureDetector_AGAST_5_8,
            cv2.AgastFeatureDetector_AGAST_7_12d,
            cv2.AgastFeatureDetector_OAST_9_16
        ],  # AGAST detector variants
        'nonmaxSuppression': [True, False]           # Enable/disable non-maximum suppression
    },
    'KAZE': {
        'threshold': [0.0005, 0.001, 0.002, 0.004],  # Detector response threshold
        'nOctaves': [2, 3, 4, 6],                   # Number of octaves
        'nOctaveLayers': [2, 3, 4, 6],              # Layers per octave
        'diffusivity': [cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_WEICKERT]  # Diffusion type
    },
    'AKAZE': {
        'threshold': [0.0005, 0.001, 0.002, 0.004],  # Detector response threshold
        'nOctaves': [3, 4, 5, 6],                   # Number of octaves
        'nOctaveLayers': [2, 3, 4, 6],              # Layers per octave
        'diffusivity': [cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2]  # Diffusion type
    }
}

PARAM_REPORT = [
    "\\begin{table}[h]",
    "\\centering",
    "\\small",
    "\\caption{Optimal Parameters for Corner Detection Algorithms}",
    "\\label{tab:optimal_parameters}",
    "\\begin{tabular}{lp{3.5cm}c}",
    "\\toprule",
    "\\textbf{Algorithm} & \\textbf{Optimal Parameters} & \\textbf{$\\#$ Tests} \\\\",
    "\\midrule"
]