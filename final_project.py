## import standard libraries
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy
import argparse
import time

## import custom scripts
import corner_methods as cm
from parameter_setup import ALGORITHMS, PARAM_GRIDS, image_path, ground_truth_path, output_path
import utilities as utils

def get_ground_truth(image_path, ground_truth_path, output_path, plot=True, images=None, image_names=None, ground_truth_corners=None):
    
    images = images or []
    image_names = image_names or []
    ground_truth_corners = ground_truth_corners or []
    
    ## read in images and truth points
    for image in sorted(os.listdir(image_path)):
        corners = f"{image.split('.')[0]}.txt"
        image_names.append(image.split('.')[0])
        images.append(cv2.imread(os.path.join(image_path, image), cv2.IMREAD_GRAYSCALE))
        
        with open(os.path.join(ground_truth_path, corners), 'r') as file:
            coordinate_pairs = np.array([tuple(map(int, line.split())) for line in file])
        ground_truth_corners.append(coordinate_pairs)
        
    if plot:
        utils.plot_ground_truth(images, image_names, ground_truth_corners, output_path)
    
    return images, image_names, ground_truth_corners
    
    
def run_benchmark(images, image_names, ground_truth_corners, output_path):
    """Run benchmarking for corner detection algorithms.
    
    Args:
        images (list): List of grayscale images.
        image_names (list): List of image names.
        ground_truth_corners (list): List of ground truth corner coordinates.
        output_path (str): Path to save results.
    
    Returns:
        dict: Results dictionary with metrics for each algorithm.
    """
    results = {name: {'speed': [], 'precision': [], 'recall': [], 'repeatability': [], 'scale_invariance': []} for name in ALGORITHMS}
    
    for img_idx, (image, gt_corners, name) in enumerate(zip(images, ground_truth_corners, image_names)):
        scaled_images = utils.create_scaled_images(image)
        
        for alg_name, alg_func in ALGORITHMS.items():
            # Optimize parameters on first image
            if img_idx == 0:
                best_params = utils.optimize_parameters(alg_func, alg_name, image, gt_corners, PARAM_GRIDS)
            else:
                # Use default parameters (first value from each parameter list)
                best_params = {k: v[0] for k, v in PARAM_GRIDS[alg_name].items()}
            
            # Test across scales
            for scale_idx, scaled_img in enumerate(scaled_images):
                start_time = time.time()
                corners = alg_func(scaled_img, args=best_params)
                exec_time = time.time() - start_time
                
                # Adjust ground truth for scale
                scaled_gt = gt_corners * (0.5 if scale_idx == 0 else 2.0 if scale_idx == 2 else 1.0)
                
                precision, recall, repeatability = utils.calculate_metrics(corners, scaled_gt)
                
                results[alg_name]['speed'].append(exec_time)
                results[alg_name]['precision'].append(precision)
                results[alg_name]['recall'].append(recall)
                results[alg_name]['repeatability'].append(repeatability)
                results[alg_name]['scale_invariance'].append(repeatability if scale_idx != 1 else 0)

    # Save results
    os.makedirs(output_path, exist_ok=True)
    for alg_name in results:
        np.savez(os.path.join(output_path, f'{alg_name}_results.npz'), **results[alg_name])
    
    return results
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark corner detection algorithms.')
    parser.add_argument('--noplot', action='store_false', dest='plot', help='Disable plotting of ground truth images.')
    args = parser.parse_args()

    np.random.seed(11001)
    images, image_names, ground_truth_corners = get_ground_truth(image_path, ground_truth_path, output_path, plot=args.plot)
    results = run_benchmark(images, image_names, ground_truth_corners, output_path)
    if args.plot:
        utils.visualize_results(ALGORITHMS, output_path)
    