## import standard libraries
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy
import argparse
import time
import itertools
from tqdm import tqdm

## import custom scripts
import corner_methods as cm
from parameters import ALGORITHMS, PARAM_GRIDS, image_path, ground_truth_path, output_path
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
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        utils.plot_ground_truth(images, image_names, ground_truth_corners, output_path)
    
    return images, image_names, ground_truth_corners

def optimize_across_images(algorithm, alg_name, images, gt_corners_list, param_grid):
    """Optimize parameters across all images for best combined accuracy."""
    best_params = {}
    best_score = -1
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        total_precision = 0
        total_recall = 0
        total_repeatability = 0
        num_images = len(images)
        
        for image, gt_corners in zip(images, gt_corners_list):
            corners = algorithm(image, args=params)
            precision, recall, repeatability = utils.calculate_metrics(corners, gt_corners)
            total_precision += precision
            total_recall += recall
            total_repeatability += repeatability
        
        avg_precision = total_precision / num_images
        avg_recall = total_recall / num_images
        avg_repeatability = total_repeatability / num_images
        score = (avg_precision + avg_recall + avg_repeatability) / 3
        
        if score > best_score:
            best_score = score
            best_params = params
    
    # Calculate total combinations tested
    total_combinations = np.prod([len(values) for values in param_values])
    
    return best_params, total_combinations

def run_benchmark(images, image_names, ground_truth_corners, output_path):
    """Run benchmarking for corner detection algorithms with optimized parameters."""
    results = {name: {'speed': [], 'precision': [], 'recall': [], 'repeatability': [], 'scale_invariance': []} for name in ALGORITHMS}
    optimized_params = {}
    
    # Optimize parameters and benchmark for each algorithm
    param_report = [
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
    
    for alg_name, alg_func in tqdm(ALGORITHMS.items()):
        # Optimize parameters across all images
        best_params, total_combinations = optimize_across_images(
            alg_func, alg_name, images, ground_truth_corners, PARAM_GRIDS[alg_name]
        )
        optimized_params[alg_name] = best_params
        
        # Format parameters with newlines between them
        param_str = "\\\\".join([f"{key.replace('_', '\\_')}: {value}" for key, value in best_params.items()])
        param_str2 = f"{alg_name} & \\multirow{{" + f"{len(best_params.items())}" + f"}}{{*}}{{\\parbox{{3.5cm}}{{\\raggedright {param_str}}}}} & \\multirow{{" + f"{len(best_params.items())}" + f"}}{{*}}{{{total_combinations}}} \\\\"
        param_report.append(param_str2)
        param_report.append("& & \\\\")  # Empty row for multirow spanning
        if len(best_params.items()) > 2:
            for i in range(len(best_params.items())-2):
                param_report.append("& & \\\\")
        
        # Benchmark with optimized parameters for this algorithm
        for img_idx, (image, gt_corners, name) in enumerate(zip(images, ground_truth_corners, image_names)):
            scaled_images = utils.create_scaled_images(image)
            
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
    
    # Finalize parameter report
    param_report.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    # Save parameter report
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "optimal_parameters.txt"), 'w') as f:
        f.write("\n".join(param_report))
    
    # Generate sample detection images with optimized parameters
    utils.generate_sample_detections(images, image_names, ground_truth_corners, optimized_params, ALGORITHMS, output_path)
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    for alg_name in results:
        np.savez(os.path.join(output_path, f'{alg_name}_results.npz'), **results[alg_name])
    
    # Generate comparison tables
    utils.generate_comparison_tables(results, output_path)
    
    return results, optimized_params

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark corner detection algorithms.')
    parser.add_argument('--noplot', action='store_false', dest='plot', help='Disable plotting of ground truth images.')
    args = parser.parse_args()

    np.random.seed(11001)
    images, image_names, ground_truth_corners = get_ground_truth(image_path, ground_truth_path, output_path, plot=args.plot)
    results, optimized_params = run_benchmark(images, image_names, ground_truth_corners, output_path)
    if args.plot:
        utils.visualize_results(ALGORITHMS, output_path)