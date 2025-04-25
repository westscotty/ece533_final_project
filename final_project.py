## import standard libraries
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import time
import itertools
from tqdm import tqdm
import re

## import custom scripts
import corner_methods as cm
from parameters import ALGORITHMS, PARAM_GRIDS, image_path, ground_truth_path, output_path, SCALES
import utilities as utils

def get_ground_truth(image_path, ground_truth_path, output_path, images=None, image_names=None, ground_truth_corners=None):
    
    images = images or []
    image_names = image_names or []
    ground_truth_corners = ground_truth_corners or []
    
    sorted_files = sorted(os.listdir(image_path), key=lambda x: int(re.search(r'\d+', x).group()))
    
    ## read in images and truth points
    for image in sorted_files:
        corners = f"{image.split('.')[0]}.txt"
        image_names.append(image.split('.')[0])
        images.append(cv2.imread(os.path.join(image_path, image), cv2.IMREAD_GRAYSCALE))
        
        with open(os.path.join(ground_truth_path, corners), 'r') as file:
            coordinate_pairs = np.array([tuple(map(int, line.split())) for line in file])
        ground_truth_corners.append(coordinate_pairs)
        

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    utils.plot_ground_truth(images, image_names, ground_truth_corners, output_path)
    
    return images, image_names, ground_truth_corners

def optimize_across_images(algorithm, images, gt_corners_list, param_grid):

    best_params = {}
    best_score = -1
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    total_combinations = np.prod([len(values) for values in param_values])
    
    # Store metrics for each parameter combination
    all_metrics = []
    
    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        speeds = []
        precisions = []
        recalls = []
        repeatabilities = []
        f_scores = []
        aprs = []
        localization_errors = []
        corner_quantities = []
        
        for image, gt_corners in zip(images, gt_corners_list):
            start_time = time.time()
            corners = algorithm(image, args=params)
            exec_time = time.time() - start_time
            precision, recall, repeatability, f_score, apr, localization_error, corner_quantity = utils.calculate_metrics(corners, gt_corners)
            
            speeds.append(exec_time)
            precisions.append(precision)
            recalls.append(recall)
            repeatabilities.append(repeatability)
            f_scores.append(f_score)
            aprs.append(apr)
            localization_errors.append(localization_error)
            corner_quantities.append(corner_quantity)
            
        # Normalize speed to [0, 1] for scoring (lower speed is better, so invert it)
        max_speed = max(speeds) if speeds else 1.0
        normalized_speeds = [1 - (speed / max_speed) if max_speed > 0 else 1.0 for speed in speeds]
        
        # Normalize localization error to [0, 1] for scoring (lower is better, so invert it)
        max_le = max(localization_errors) if localization_errors and max(localization_errors) > 0 else 1.0
        normalized_le = [1 - (le / max_le) if max_le > 0 else 1.0 for le in localization_errors]
        
        # Compute average metrics
        avg_speed = np.mean(normalized_speeds)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_repeatability = np.mean(repeatabilities)
        avg_f_score = np.mean(f_scores)
        avg_apr = np.mean(aprs)
        avg_le = np.mean(normalized_le)
        
        # Score is the average of the normalized metrics (excluding corner_quantity since it's not part of optimization)
        score = (avg_speed + avg_precision + avg_recall + avg_repeatability + avg_f_score + avg_apr + avg_le) / 7
        
        # Store metrics for this combination
        all_metrics.append({
            'params': params,
            'speed': speeds,
            'precision': precisions,
            'recall': recalls,
            'repeatability': repeatabilities,
            'f_score': f_scores,
            'apr': aprs,
            'localization_error': localization_errors,
            'corner_quantity': corner_quantities,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, total_combinations, all_metrics

def test_scale_invariance(images, image_names, ground_truth_corners, optimized_params):

    scale_results = {name: {img_name: {
        'speed': [], 'precision': [], 'recall': [], 'repeatability': [], 'f_score': [], 'apr': [],
        'localization_error': [], 'corner_quantity': [], 'scale_invariance': 0.0
    } for img_name in image_names} for name in ALGORITHMS}
        
    # Select a random image for sample visualization
    sample_idx = np.random.randint(0, len(images))
    sample_image = images[sample_idx]
    sample_name = image_names[sample_idx]
    sample_gt_corners = ground_truth_corners[sample_idx]
    
    for alg_name, alg_func in tqdm(ALGORITHMS.items(), desc="Testing Scale Invariance"):
        best_params = optimized_params[alg_name]
        scaled_images = utils.create_scaled_images(sample_image, SCALES)
        sample_corners = []
        
        for img_idx, (image, gt_corners, name) in enumerate(zip(images, ground_truth_corners, image_names)):
            scaled_images_current = utils.create_scaled_images(image, SCALES)
            repeatabilities = []  # To compute scale invariance
            
            for scale_idx, scaled_img in enumerate(scaled_images_current):
                start_time = time.time()
                corners = alg_func(scaled_img, args=best_params)
                exec_time = time.time() - start_time
                
                # Adjust ground truth for scale
                scaled_gt = gt_corners * SCALES[scale_idx]
                
                precision, recall, repeatability, f_score, apr, localization_error, corner_quantity = utils.calculate_metrics(corners, scaled_gt)
                
                # Store results for this scale
                scale_results[alg_name][name]['speed'].append(exec_time)
                scale_results[alg_name][name]['precision'].append(precision)
                scale_results[alg_name][name]['recall'].append(recall)
                scale_results[alg_name][name]['repeatability'].append(repeatability)
                scale_results[alg_name][name]['f_score'].append(f_score)
                scale_results[alg_name][name]['apr'].append(apr)
                scale_results[alg_name][name]['localization_error'].append(localization_error)
                scale_results[alg_name][name]['corner_quantity'].append(corner_quantity)
                
                repeatabilities.append(repeatability)
                
                # Store corners for the sample image
                if img_idx == sample_idx:
                    sample_corners.append(corners)
            
            # Compute scale invariance as 1 - coefficient of variation of repeatability
            repeatabilities = np.array(repeatabilities)
            if repeatabilities.mean() > 0:
                cv = repeatabilities.std() / repeatabilities.mean()
                scale_invariance = max(0, min(1, 1 - cv))  # Higher value = better scale invariance
            else:
                scale_invariance = 0.0
            
            scale_results[alg_name][name]['scale_invariance'] = scale_invariance
        
        # Generate sample visualization for this algorithm
        utils.generate_scale_invariance_samples(
            scaled_images, sample_corners, sample_gt_corners, SCALES, 
            alg_name, sample_name, output_path
        )
    
    return scale_results

def run_benchmarking(images, image_names, ground_truth_corners):

    results = {name: {
        'speed': [], 'precision': [], 'recall': [], 'repeatability': [], 'f_score': [], 'apr': [],
        'localization_error': [], 'corner_quantity': []
    } for name in ALGORITHMS}
    individual_results = {name: {img_name: {
        'speed': [], 'precision': [], 'recall': [], 'repeatability': [], 'f_score': [], 'apr': [],
        'localization_error': [], 'corner_quantity': []
    } for img_name in image_names} for name in ALGORITHMS}
    optimized_params = {}
    all_metrics_per_algorithm = {}  # Store all metrics for each parameter combination per algorithm
    
    # Optimize parameters and benchmark for each algorithm
    params = []
    combinations = []
    
    for alg_name, alg_func in tqdm(ALGORITHMS.items(), desc="Generating Optimization Data"):
        # Optimize parameters across all images and store all metrics
        best_params, total_combinations, all_metrics = optimize_across_images(alg_func, images, ground_truth_corners, PARAM_GRIDS[alg_name])
        params.append(best_params)
        combinations.append(total_combinations)
        optimized_params[alg_name] = best_params
        all_metrics_per_algorithm[alg_name] = all_metrics
        
        # Benchmark with optimized parameters for this algorithm
        for img_idx, (image, gt_corners, name) in enumerate(zip(images, ground_truth_corners, image_names)):
            start_time = time.time()
            corners = alg_func(image, args=best_params)
            exec_time = time.time() - start_time
            
            precision, recall, repeatability, f_score, apr, localization_error, corner_quantity = utils.calculate_metrics(corners, gt_corners)
            
            # Store aggregated results
            results[alg_name]['speed'].append(exec_time)
            results[alg_name]['precision'].append(precision)
            results[alg_name]['recall'].append(recall)
            results[alg_name]['repeatability'].append(repeatability)
            results[alg_name]['f_score'].append(f_score)
            results[alg_name]['apr'].append(apr)
            results[alg_name]['localization_error'].append(localization_error)
            results[alg_name]['corner_quantity'].append(corner_quantity)
            
            # Store individual results
            individual_results[alg_name][name]['speed'].append(exec_time)
            individual_results[alg_name][name]['precision'].append(precision)
            individual_results[alg_name][name]['recall'].append(recall)
            individual_results[alg_name][name]['repeatability'].append(repeatability)
            individual_results[alg_name][name]['f_score'].append(f_score)
            individual_results[alg_name][name]['apr'].append(apr)
            individual_results[alg_name][name]['localization_error'].append(localization_error)
            individual_results[alg_name][name]['corner_quantity'].append(corner_quantity)
    
    return results, optimized_params, individual_results, params, combinations, all_metrics_per_algorithm

if __name__ == "__main__":

    np.random.seed(11001)
    print("Loading Ground Truth ...")
    images, image_names, ground_truth_corners = get_ground_truth(image_path, ground_truth_path, output_path)
    results, optimized_params, individual_results, params, combinations, all_metrics_per_algorithm = run_benchmarking(images, image_names, ground_truth_corners)
    
    # Run scale invariance test
    scale_results = test_scale_invariance(images, image_names, ground_truth_corners, optimized_params)
        
    # Generate sample detection images with optimized parameters
    print("Creating Sample Imagery for Optimized Algorithms ...")
    utils.generate_sample_detections(images, image_names, ground_truth_corners, optimized_params, ALGORITHMS, output_path)
    
    # Save results
    print("Writing Results Files ...")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "data"), exist_ok=True)
    for alg_name in results:
        np.savez(os.path.join(output_path, f'data/{alg_name}_results.npz'), **results[alg_name])
        for img_name in image_names:
            np.savez(os.path.join(output_path, f'data/{alg_name}_{img_name}_individual_results.npz'), **individual_results[alg_name][img_name])
    
    # Generate comparison tables and reports
    utils.generate_param_report(params, combinations, output_path)
    utils.generate_comparison_tables(results, output_path)
    
    # Generate individual result plots and pairwise metric plots
    print("Plotting Result Images ...")
    utils.visualize_results(ALGORITHMS, output_path)
    utils.plot_pairwise_metrics(results, output_path)
    utils.plot_best_combination(individual_results, image_names, output_path)
    utils.plot_all_combinations(all_metrics_per_algorithm, image_names, output_path)
    utils.plot_scale_invariance(scale_results, image_names, output_path)