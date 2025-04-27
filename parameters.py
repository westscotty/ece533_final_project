from corner_methods import *
import os
from distributions import UniformDist, NormalDist, CategoricalDist

image_path = os.path.join(os.getcwd(), "data/Urban_Corner_datasets/Images")
ground_truth_path = os.path.join(os.getcwd(), "data/Urban_Corner_datasets/Ground_Truth")
output_path = os.path.join(os.getcwd(), "results")

SCALES = [0.25, 1.0, 4.0]  # Scales to test

N_SAMPLES = 20
MAX_SAMPLES = 250
SEED = 11003

ALGORITHMS = {
                'Harris'    : harris,
                'Shi-Tomasi': shi_tomasi,
                'FAST'      : fast,
                'ORB'       : orb,
                'SIFT'      : sift,
                'BRISK'     : brisk,
                'AGAST'     : agast,
                'KAZE'      : kaze,
                'AKAZE'     : akaze
            }

## Using custom distribution generation code
PARAM_GRIDS = {
    'Harris': {
        'blockSize': UniformDist(min_val=2, max_val=11, is_int=True),
        'ksize': CategoricalDist(options=[3, 5, 7, 9, 11]),  # Only odd integers
        'k': UniformDist(min_val=0.01, max_val=0.15),
        'borderType': CategoricalDist(options=[
            cv2.BORDER_DEFAULT, cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT
        ])
    },
    # ... (rest of PARAM_GRIDS unchanged)
    'Shi-Tomasi': {
        'maxCorners': UniformDist(min_val=25, max_val=1000, is_int=True),
        'qualityLevel': UniformDist(min_val=0.005, max_val=0.2),
        'minDistance': UniformDist(min_val=3, max_val=30, is_int=True),
        'blockSize': UniformDist(min_val=3, max_val=9, is_int=True)
    },
    'FAST': {
        'threshold': UniformDist(min_val=5, max_val=100, is_int=True),
        'type': CategoricalDist(options=[
            cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
            cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
            cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        ])
    },
    'ORB': {
        'nfeatures': UniformDist(min_val=100, max_val=5000, is_int=True),
        'scaleFactor': UniformDist(min_val=1.05, max_val=1.5),
        'nlevels': UniformDist(min_val=4, max_val=16, is_int=True),
        'edgeThreshold': UniformDist(min_val=15, max_val=70, is_int=True),
        'patchSize': UniformDist(min_val=15, max_val=50, is_int=True),
        'fastThreshold': UniformDist(min_val=5, max_val=100, is_int=True)
    },
    'SIFT': {
        'nOctaveLayers': UniformDist(min_val=2, max_val=6, is_int=True),
        'contrastThreshold': UniformDist(min_val=0.02, max_val=0.16),
        'edgeThreshold': UniformDist(min_val=5, max_val=30, is_int=True),
        'sigma': NormalDist(mean=1.6, std=0.4, min_val=1.0, max_val=2.4)
    },
    'BRISK': {
        'thresh': UniformDist(min_val=10, max_val=100, is_int=True),
        'octaves': UniformDist(min_val=2, max_val=6, is_int=True),
        'patternScale': UniformDist(min_val=0.5, max_val=2.0)
    },
    'AGAST': {
        'threshold': UniformDist(min_val=5, max_val=50, is_int=True),
        'type': CategoricalDist(options=[
            cv2.AgastFeatureDetector_AGAST_5_8,
            cv2.AgastFeatureDetector_AGAST_7_12d,
            cv2.AgastFeatureDetector_OAST_9_16
        ])
    },
    'KAZE': {
        'threshold': UniformDist(min_val=0.0005, max_val=0.004),
        'nOctaves': UniformDist(min_val=2, max_val=6, is_int=True),
        'nOctaveLayers': UniformDist(min_val=2, max_val=6, is_int=True),
        'diffusivity': CategoricalDist(options=[
            cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2, cv2.KAZE_DIFF_WEICKERT
        ])
    },
    'AKAZE': {
        'threshold': UniformDist(min_val=0.0005, max_val=0.004),
        'nOctaves': UniformDist(min_val=3, max_val=6, is_int=True),
        'nOctaveLayers': UniformDist(min_val=2, max_val=6, is_int=True),
        'diffusivity': CategoricalDist(options=[
            cv2.KAZE_DIFF_PM_G1, cv2.KAZE_DIFF_PM_G2
        ])
    }
}

PARAM_REPORT = [
    "\\begin{table}[h]",
    "\\centering",
    "\\small",
    "\\caption{Optimal Parameters for Corner Detection Algorithms}",
    "\\label{tab:optimal_parameters}",
    "\\begin{tabular}{lp{3.5cm}cc}",
    "\\toprule",
    "\\textbf{Algorithm} & \\textbf{Optimal Parameters} & \\textbf{$\\#$ Tests} & \\textbf{Best Score} \\\\",
    "\\midrule"
]