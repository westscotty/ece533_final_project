from corner_methods import *
import os

image_path = os.path.join(os.getcwd(), "data/Urban_Corner_datasets/Images")
ground_truth_path = os.path.join(os.getcwd(), "data/Urban_Corner_datasets/Ground_Truth")
output_path = os.path.join(os.getcwd(), "results")

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

# Parameter grids for optimization
PARAM_GRIDS = {
    'Harris': {
        'blockSize': [2, 3, 5],
        'ksize': [3, 5, 7],
        'k': [0.04, 0.06, 0.08]
    },
    'Shi-Tomasi': {
        'maxCorners': [25, 50, 100],
        'qualityLevel': [0.01, 0.05, 0.1],
        'minDistance': [5, 10, 20]
    },
    'FAST': {
        'threshold': [10, 25, 50],
        'type': [cv2.FAST_FEATURE_DETECTOR_TYPE_5_8, cv2.FAST_FEATURE_DETECTOR_TYPE_7_12, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16]
    },
    'ORB': {
        'nfeatures': [500, 1000],
        'scaleFactor': [1.1, 1.2],
        'nlevels': [8, 10]
    },
    'SIFT': {
        'nOctaveLayers': [3, 4],
        'contrastThreshold': [0.04, 0.08],
        'edgeThreshold': [10, 20]
    },
    'BRISK': {
        'thresh': [30, 50],
        'octaves': [3, 4]
    },
    'AGAST': {
        'threshold': [10, 20],
        'type': [0, 1, 2]
    },
    'KAZE': {
        'threshold': [0.001, 0.002],
        'nOctaves': [3, 4]
    },
    'AKAZE': {
        'threshold': [0.001, 0.002],
        'nOctaves': [4, 5]
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