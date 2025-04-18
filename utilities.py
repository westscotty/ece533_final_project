import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score
from itertools import product
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist

def plot_ground_truth(images, image_names, ground_truth_corners, output_path):
    
    plt.figure(figsize=(8,10))
    plt.suptitle("Ground Truth Images")
    for i in range(len(images)):
        plt.subplot(6, 4, i+1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
        plt.title(f"Image {image_names[i]}")
        for corner in ground_truth_corners[i]:
            plt.scatter(corner[1], corner[0], color="red", marker='o', s=2)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ground_truth.png"))
    plt.show()

def create_scaled_images(image, scales=[0.5, 1.0, 2.0]):
    scaled_images = []
    for scale in scales:
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_images.append(scaled)
    return scaled_images


# def calculate_metrics(pred_corners, gt_corners, threshold=5.0):
#     if len(gt_corners) == 0:
#         return 0.0, 0.0, 0.0

#     gt = gt_corners.astype(np.float32)
#     pred = pred_corners.astype(np.float32)

#     dists = cdist(gt, pred)  # shape: (num_gt, num_pred)
#     matched_gt = np.any(dists <= threshold, axis=1)
#     matched_pred = np.any(dists <= threshold, axis=0)

#     TP = np.sum(matched_gt)
#     FP = len(pred) - np.sum(matched_pred)
#     FN = len(gt) - TP

#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#     repeatability = TP / len(gt) if len(gt) > 0 else 0

#     return precision, recall, repeatability

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

def calculate_metrics(pred_corners, gt_corners, threshold=5.0):
    gt = np.array(gt_corners, dtype=np.float32)
    pred = np.array(pred_corners, dtype=np.float32)

    # Reshape if necessary
    if pred.ndim == 1 and pred.shape[0] == 2:
        pred = pred.reshape(1, 2)
    if pred.ndim == 1 or pred.size == 0:
        pred = np.empty((0, 2), dtype=np.float32)

    if gt.ndim == 1 and gt.shape[0] == 2:
        gt = gt.reshape(1, 2)
    if gt.ndim == 1 or gt.size == 0:
        gt = np.empty((0, 2), dtype=np.float32)

    # Early return if either side has no corners
    if len(gt) == 0 or len(pred) == 0:
        return 0.0, 0.0, 0.0

    dists = cdist(gt, pred)  # shape: (num_gt, num_pred)
    matched_gt = np.any(dists <= threshold, axis=1)
    matched_pred = np.any(dists <= threshold, axis=0)

    TP = np.sum(matched_gt)
    FP = len(pred) - np.sum(matched_pred)
    FN = len(gt) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    repeatability = TP / len(gt) if len(gt) > 0 else 0

    return precision, recall, repeatability

def optimize_parameters(algorithm, alg_name, image, gt_corners, PARAM_GRIDS):
    best_params = {}
    best_score = -1
    param_grid = PARAM_GRIDS.get(alg_name, {})
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        
        # Call the algorithm wrapper with parameters
        corners = algorithm(image, args=params)
        
        precision, recall, repeatability = calculate_metrics(corners, gt_corners)
        score = (precision + recall + repeatability) / 3
    
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params

def visualize_results(ALGORITHMS, output_path):
    plt.figure(figsize=(15, 10))
    
    for idx, metric in enumerate(['speed', 'precision', 'recall', 'repeatability', 'scale_invariance']):
        plt.subplot(2, 3, idx + 1)
        for alg_name in ALGORITHMS:
            data = np.load(os.path.join(output_path, f'{alg_name}_results.npz'))[metric]
            plt.plot(data, label=alg_name)
        plt.title(metric.capitalize())
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'benchmark_results.png'))
    plt.close()
