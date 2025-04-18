import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from itertools import product
from corner_methods import harris, shi_tomasi, fast, orb, sift, brisk, agast, kaze, akaze

# Configuration
DATASET_PATH = "data/Urban_Corner_datasets"
IMAGE_PATH = os.path.join(DATASET_PATH, "Images")
GROUND_TRUTH_PATH = os.path.join(DATASET_PATH, "Ground_Truth")
OUTPUT_PATH = "results"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Algorithms to benchmark
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

def load_dataset():
    images = []
    image_names = []
    ground_truth_corners = []
    
    for image_file in sorted(os.listdir(IMAGE_PATH)):
        if not image_file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(IMAGE_PATH, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        image_names.append(image_file.split('.')[0])
        
        gt_file = f"{image_names[-1]}.txt"
        gt_path = os.path.join(GROUND_TRUTH_PATH, gt_file)
        with open(gt_path, 'r') as file:
            corners = [tuple(map(int, line.split())) for line in file]
            ground_truth_corners.append(np.array(corners, dtype=np.float32))
    
    return images, image_names, ground_truth_corners

# def create_scaled_images(image, scales=[0.5, 1.0, 2.0]):
#     scaled_images = []
#     for scale in scales:
#         scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#         scaled_images.append(scaled)
#     return scaled_images

# def calculate_metrics(detected_corners, gt_corners, threshold=5):
#     if len(detected_corners) == 0 or len(gt_corners) == 0:
#         return 0.0, 0.0, 0.0
    
#     # Convert to same format
#     detected = detected_corners.astype(np.float32)
#     gt = gt_corners.astype(np.float32)
    
#     # Create matches
#     matches = []
#     for det in detected:
#         distances = np.sqrt(((gt - det) ** 2).sum(axis=1))
#         if len(distances) > 0 and np.min(distances) < threshold:
#             matches.append(1)
#         else:
#             matches.append(0)
    
#     precision = precision_score([1] * len(gt), matches, zero_division=0)
#     recall = recall_score([1] * len(gt), matches, zero_division=0)
#     repeatability = len([m for m in matches if m == 1]) / max(len(gt), 1)
    
#     return precision, recall, repeatability

# def optimize_parameters(algorithm, alg_name, image, gt_corners):
#     best_params = {}
#     best_score = -1
#     param_grid = PARAM_GRIDS.get(alg_name, {})
    
#     # Generate all parameter combinations
#     param_names = list(param_grid.keys())
#     param_values = list(param_grid.values())
#     for combo in product(*param_values):
#         params = dict(zip(param_names, combo))
        
#         # Update algorithm parameters
#         if alg_name == 'Harris':
#             corners = cv2.cornerHarris(image, **params)
#             corners = cv2.dilate(corners, None)
#             corners = np.column_stack(np.where(corners > 0.01 * corners.max()))[:, ::-1]
#         elif alg_name == 'Shi-Tomasi':
#             corners = cv2.goodFeaturesToTrack(image, **params1, useGradient=False, **params)
#             corners = corners[:, 0] if corners is not None else np.array([])
#         else:
#             detector = algorithm(image, args=params)
#             corners = detector

#         precision, recall, repeatability = calculate_metrics(corners, gt_corners)
#         score = (precision + recall + repeatability) / 3
    
#         if score > best_score:
#             best_score = score
#             best_params = params
    
#     return best_params

def run_benchmark():
    images, image_names, ground_truth_corners = load_dataset()
    results = {name: {'speed': [], 'precision': [], 'recall': [], 'repeatability': [], 'scale_invariance': []} for name in ALGORITHMS}
    
    for img_idx, (image, gt_corners, name) in enumerate(zip(images, ground_truth_corners, image_names)):
        scaled_images = create_scaled_images(image)
        
        for alg_name, alg_func in ALGORITHMS.items():
            # Optimize parameters on first image
            if img_idx == 0:
                best_params = optimize_parameters(alg_func, alg_name, image, gt_corners)
            else:
                best_params = PARAM_GRIDS[alg_name]  # Use default for subsequent images
            
            # Test across scales
            for scale_idx, scaled_img in enumerate(scaled_images):
                start_time = time.time()
                if alg_name == 'Harris':
                    corners = cv2.cornerHarris(scaled_img, **best_params)
                    corners = cv2.dilate(corners, None)
                    corners = np.column_stack(np.where(corners > 0.01 * corners.max()))[:, ::-1]
                elif alg_name == 'Shi-Tomasi':
                    corners = cv2.goodFeaturesToTrack(scaled_img, **best_params)
                    corners = corners[:, 0] if corners is not None else np.array([])
                else:
                    corners = alg_func(scaled_img, args=best_params)
                exec_time = time.time() - start_time
                
                # Adjust ground truth for scale
                scaled_gt = gt_corners * (0.5 if scale_idx == 0 else 2.0 if scale_idx == 2 else 1.0)
                
                precision, recall, repeatability = calculate_metrics(corners, scaled_gt)
                
                results[alg_name]['speed'].append(exec_time)
                results[alg_name]['precision'].append(precision)
                results[alg_name]['recall'].append(recall)
                results[alg_name]['repeatability'].append(repeatability)
                results[alg_name]['scale_invariance'].append(repeatability if scale_idx != 1 else 0)

    # Save results
    for alg_name in results:
        np.savez(os.path.join(OUTPUT_PATH, f'{alg_name}_results.npz'), **results[alg_name])

def visualize_results():
    plt.figure(figsize=(15, 10))
    
    for idx, metric in enumerate(['speed', 'precision', 'recall', 'repeatability', 'scale_invariance']):
        plt.subplot(2, 3, idx + 1)
        for alg_name in ALGORITHMS:
            data = np.load(os.path.join(OUTPUT_PATH, f'{alg_name}_results.npz'))[metric]
            plt.plot(data, label=alg_name)
        plt.title(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'benchmark_results.png'))
    plt.close()

def main():
    run_benchmark()
    visualize_results()
    print(f"Results saved in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()