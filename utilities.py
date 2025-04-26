import numpy as np
import cv2
from itertools import product
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import cdist
from tqdm import tqdm
from parameters import PARAM_REPORT, ALGORITHMS, SCALES


def plot_ground_truth(images, image_names, ground_truth_corners, output_path):
    
    plt.figure(figsize=(8,10))
    plt.suptitle("Ground Truth Images")
    for i in range(len(images)):
        plt.subplot(6, 4, i+1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")
        plt.title(f"Image {image_names[i]}")
        for corner in ground_truth_corners[i]:
            plt.scatter(corner[1], corner[0], color="blue", marker='o', s=10)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "ground_truth.png"))
    plt.close()


def create_scaled_images(image):
    scaled_images = []
    for scale in SCALES:
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_images.append(scaled)
    return scaled_images


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

    # Overall corner quantity (total number of detected corners)
    corner_quantity = len(pred)
    corner_quantity_ratio = 1 / (np.abs(len(pred) - len(gt_corners)) + 1) ## 1/(|gt - pred| + 1)
    # Early return if either side has no corners
    if len(gt) == 0 or len(pred) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, corner_quantity, corner_quantity_ratio

    # Compute distances between ground truth and predicted corners
    dists = cdist(gt, pred)  # shape: (num_gt, num_pred)
    matched_gt = np.any(dists <= threshold, axis=1)
    matched_pred = np.any(dists <= threshold, axis=0)

    TP = np.sum(matched_gt)
    FP = len(pred) - np.sum(matched_pred)
    FN = len(gt) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    repeatability = TP / len(gt) if len(gt) > 0 else 0

    denom = (precision + recall)
    f_score = 2 * (precision * recall) / denom if denom > 0 else 0
    apr = (precision + recall) / 2  # Arithmetic Mean of Precision and Recall

    # Compute localization error for matched corners
    localization_error = 0.0
    if TP > 0:
        # Find the closest predicted corner for each ground truth corner
        min_dists = np.min(dists, axis=1)  # Minimum distance for each ground truth corner
        matched_dists = min_dists[matched_gt]  # Distances for matched ground truth corners
        localization_error = np.mean(matched_dists) if len(matched_dists) > 0 else 0.0

    return precision, recall, repeatability, f_score, apr, localization_error, corner_quantity, corner_quantity_ratio

        
def generate_sample_detections(images, image_names, ground_truth_corners, optimized_params, algorithms, output_path):

    sample_output_path = os.path.join(output_path, "sample_detections")
    os.makedirs(sample_output_path, exist_ok=True)
    
    idx = np.random.randint(0, len(images))
    image = images[idx]
    name = image_names[idx]
    
    for alg_name, alg_func in algorithms.items():
        params = optimized_params[alg_name]
        corners = alg_func(image, args=params)
        gt_corners = ground_truth_corners[idx]
        
        plt.figure(figsize=(4, 8))  # Adjusted for 1x2 grid
        plt.suptitle(f"Sample Detection: {alg_name} (Image {name})", fontsize=12)
        
        # Ground truth subplot (left column)
        plt.subplot(2, 1, 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Ground Truth (x{len(gt_corners)})", fontsize=10)
        plt.axis("off")
        
        # Plot ground truth corners
        for corner in gt_corners:
            plt.scatter(corner[1], corner[0], c='blue', marker='o', s=15, label='Ground Truth')
        
        # Detected corners subplot (right column)
        plt.subplot(2, 1, 2)
        plt.imshow(image, cmap="gray")
        plt.title(f"Detected (x{len(corners)})", fontsize=10)
        plt.axis("off")
        
        # Plot detected corners
        size = 15
        if len(corners) > 500:
            size = 5
        for corner in corners:
            plt.scatter(corner[0], corner[1], c='red', marker='o', s=size, label='Detected')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(sample_output_path, f"{alg_name}_detected.png"))
        plt.close()
        
        
def generate_scale_invariance_samples(scaled_images, detected_corners, gt_corners, alg_name, image_name, output_path):
    
    sample_output_path = os.path.join(output_path, "scale_invariance_samples")
    os.makedirs(sample_output_path, exist_ok=True)
    
    plt.figure(figsize=(8, 12))  # Adjusted for 3x2 grid
    plt.suptitle(f"Scale Invariance Sample: {alg_name} (Image {image_name})", fontsize=12)
    
    for i, (scale, image, corners) in enumerate(zip(SCALES, scaled_images, detected_corners)):
        # Ground truth subplot (left column, i.e., subplot 1, 3, 5)
        plt.subplot(3, 2, 2 * i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Ground Truth (x{len(gt_corners)}) (Scale {scale})", fontsize=10)
        plt.axis("off")
        
        # Plot ground truth corners (adjusted for scale)
        scaled_gt = gt_corners * scale
        for corner in scaled_gt:
            plt.scatter(corner[1], corner[0], c='blue', marker='o', s=15, label='Ground Truth' if i == 0 else "")
        
        # Detected corners subplot (right column, i.e., subplot 2, 4, 6)
        plt.subplot(3, 2, 2 * i + 2)
        plt.imshow(image, cmap="gray")
        plt.title(f"Detected (x{len(corners)}) (Scale {scale})", fontsize=10)
        plt.axis("off")
        
        size = 15
        if len(corners) > 500:
            size = 5
        
        if scale < 1:
            size = 5
        # Plot detected corners
        for corner in corners:
            plt.scatter(corner[0], corner[1], c='red', marker='o', s=size, label='Detected' if i == 0 else "")
        
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(sample_output_path, f"{alg_name}_scale_invariance_sample.png"))
    plt.close()
    

def generate_comparison_tables(results, output_path):
    metrics = ['precision', 'recall', 'repeatability', 'speed', 'f_score', 'apr', 'localization_error', 'corner_quantity_ratio']
    table_content = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Comparison of Corner Detection Algorithms}",
        "\\label{tab:corner_detection_comparison}",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "\\textbf{Algorithm} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{Repeatability} & \\textbf{Speed (s)} & \\textbf{F Score} & \\textbf{APR} & \\textbf{Localization Error} & \\textbf{Corner Quantity Ratio} \\\\",
        "\\midrule"
    ]
    
    for alg_name in results:
        row = f"{alg_name} "
        for metric in metrics:
            data = results[alg_name][metric]
            mean_value = np.mean(data) if data else 0.0
            if metric == 'speed':
                row += f"& {mean_value:.4f} "
            elif metric == 'corner_quantity':
                row += f"& {int(mean_value)} "
            else:
                row += f"& {mean_value:.3f} "
        row += "\\\\"
        table_content.append(row)
    
    table_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    table_path = os.path.join(output_path, "comparison_table.txt")
    with open(table_path, 'w') as f:
        f.write("\n".join(table_content))

def visualize_results(ALGORITHMS, output_path):
    plt.figure(figsize=(9, 9))
    
    for idx, metric in enumerate(['speed', 'precision', 'recall', 'repeatability', 'f_score', 'apr', 'localization_error', 'corner_quantity_ratio']):
        plt.subplot(3, 3, idx + 1)
        for alg_name in ALGORITHMS:
            data = np.load(os.path.join(output_path, f'data/{alg_name}_results.npz'))[metric]
            plt.plot(data, label=alg_name)
        plt.title(metric.capitalize())
        plt.legend(loc="upper right", fontsize="xx-small")
        plt.xlabel('Image Number')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'benchmark_results.png'))
    plt.close()

def plot_best_combination(individual_results, image_names, output_path):
    """Plot the metrics of the best parameter combination for each algorithm across images."""
    plot_output_path = os.path.join(output_path, "best_combination_plots")
    os.makedirs(plot_output_path, exist_ok=True)
    
    metrics = ['speed', 'precision', 'recall', 'repeatability', 'f_score', 'apr', 'localization_error', 'corner_quantity_ratio']
    
    for alg_name in individual_results:
        plt.figure(figsize=(9, 9))
        plt.suptitle(f"Best Combination Metrics for {alg_name}")
        
        for idx, metric in enumerate(metrics):
            plt.subplot(3, 3, idx + 1)
            # Gather metric data across all images
            metric_data = [individual_results[alg_name][img_name][metric][0] for img_name in image_names]
            plt.plot(image_names, metric_data, marker='o', label=alg_name)
            plt.title(metric.capitalize())
            plt.xlabel('Image Number')
            plt.ylabel(metric.capitalize())
            plt.xticks(rotation=45)
            if metric not in ['speed', 'localization_error', 'corner_quantity']:
                plt.ylim(0, 1)
            else:
                plt.ylim(0, np.max(metric_data) * 1.2 if np.max(metric_data) > 0 else 1)
            plt.legend(loc="upper right")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plot_output_path, f"{alg_name}_best_combination.png"))
        plt.close()
        

def plot_all_combinations(all_metrics_per_algorithm, image_names, output_path):
    plot_output_path = os.path.join(output_path, "all_combinations_plots")
    os.makedirs(plot_output_path, exist_ok=True)
    
    metrics = ['speed', 'precision', 'recall', 'repeatability', 'f_score', 'apr', 'localization_error', 'corner_quantity_ratio']
    
    for alg_name in all_metrics_per_algorithm:
        all_metrics = all_metrics_per_algorithm[alg_name]
        # Find the best combination (highest score)
        best_idx = np.argmax([m['score'] for m in all_metrics])
        
        plt.figure(figsize=(9, 9))
        plt.suptitle(f"All Parameter Combinations for {alg_name}")
        
        for idx, metric in enumerate(metrics):
            plt.subplot(3, 3, idx + 1)
            
            # Plot all combinations with transparency
            for combo_idx, metrics_dict in enumerate(all_metrics):
                metric_data = metrics_dict[metric]
                if combo_idx == best_idx:
                    # Highlight the best combination with a bold line
                    plt.plot(image_names, metric_data, marker='o', color='red', linewidth=2, label='Best Combination', markersize=3)
                else:
                    # Plot other combinations with transparency
                    plt.plot(image_names, metric_data, marker='o', alpha=0.3, color='blue', markersize=3, label='Other Combinations' if combo_idx == 0 else "")
            
            plt.title(metric.capitalize())
            plt.xlabel('Image Number')
            plt.ylabel(metric.capitalize())
            # plt.xticks(rotation=45)
            if metric not in ['speed', 'localization_error', 'corner_quantity']:
                plt.ylim(0, 1)
            else:
                all_values = [np.max(metrics_dict[metric]) for metrics_dict in all_metrics]
                plt.ylim(0, np.max(all_values) * 1.2 if np.max(all_values) > 0 else 1)
            plt.legend(loc="upper right")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plot_output_path, f"{alg_name}_all_combinations.png"))
        plt.close()
        

def plot_scale_invariance(scale_results, image_names, output_path):
    plot_output_path = os.path.join(output_path, "scale_invariance_plots")
    os.makedirs(plot_output_path, exist_ok=True)
    
    metrics = ['speed', 'precision', 'recall', 'repeatability', 'f_score', 'apr', 'localization_error', 'corner_quantity_ratio']
    scale_labels = [str(val) for val in SCALES]
    
    for alg_name in scale_results:
        plt.figure(figsize=(9, 9))
        # Compute average scale invariance score across all images
        avg_scale_invariance = np.mean([scale_results[alg_name][img_name]['scale_invariance'] for img_name in image_names])
        plt.suptitle(f"Scale Invariance Test for {alg_name} (Avg Scale Invariance: {avg_scale_invariance:.3f})")
        
        for idx, metric in enumerate(metrics):
            plt.subplot(3, 3, idx + 1)
            
            # Plot a line for each image across scales
            for img_idx, img_name in enumerate(image_names):
                metric_data = scale_results[alg_name][img_name][metric]
                plt.plot(scale_labels, metric_data, marker='o', label=f"Image {img_name}")
            
            plt.title(metric.capitalize())
            plt.xlabel('Scale')
            plt.ylabel(metric.capitalize())
            if metric not in ['speed', 'localization_error', 'corner_quantity']:
                plt.ylim(0, 1)
            else:
                all_values = [np.max(scale_results[alg_name][img_name][metric]) for img_name in image_names]
                plt.ylim(0, np.max(all_values) * 1.2 if np.max(all_values) > 0 else 1)
            # plt.legend(loc="upper right")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(plot_output_path, f"{alg_name}_scale_invariance.png"))
        plt.close() 
        

def generate_param_report(params, combinations, scores, output_path):
    param_report = PARAM_REPORT
    for i, (alg_name, alg_func) in enumerate(ALGORITHMS.items()):
        best_params = params[i]
        total_combinations = combinations[i]
        best_score = scores[i]
        
        param_str = "\\\\".join([f"{key.replace('_', '\\_')}: {value}" for key, value in best_params.items()])
        param_str2 = f"\\multirow{{" + f"{len(best_params.items())}" + f"}}{{*}}{{{alg_name}}} & \\multirow{{" + f"{len(best_params.items())}" + f"}}{{*}}{{\\parbox{{3.25cm}}{{\\raggedright {param_str}}}}} & \\multirow{{" + f"{len(best_params.items())}" + f"}}{{*}}{{{total_combinations}}}  & \\multirow{{" + f"{len(best_params.items())}" + f"}}{{*}}{{{np.round(best_score, 4)}}} \\\\"
        param_report.append(param_str2)
        param_report.append("& & \\\\")
        param_report.append("& & \\\\")
        param_report.append("& & \\\\")
        
        if len(best_params.items()) > 2:
            for i in range(len(best_params.items())-2):
                param_report.append("& & \\\\")
                    
    param_report.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ]) 
    with open(os.path.join(output_path, "optimal_parameters.txt"), 'w') as f:
        f.write("\n".join(param_report))
        
def generate_scale_invariance_table(scale_results, output_path):
    """Generate a LaTeX table of mean scale invariance scores for each algorithm."""
    table_output_path = os.path.join(output_path, "scale_invariance_table.txt")
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize table content
    table_content = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Mean Scale Invariance Scores for Corner Detection Algorithms}",
        r"\label{tab:scale_invariance_scores}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Algorithm & Scale Invariance Score \\",
        r"\midrule",
    ]
    
    # Compute mean scale invariance score for each algorithm
    for alg_name in sorted(scale_results.keys()):
        scores = [scale_results[alg_name][img_name]['scale_invariance'] 
                  for img_name in scale_results[alg_name]]
        mean_score = np.mean(scores)
        # Format algorithm name (e.g., convert 'ShiTomasi' to 'Shi-Tomasi')
        formatted_name = alg_name.replace('ShiTomasi', 'Shi-Tomasi')
        table_content.append(f"{formatted_name} & {mean_score:.3f} \\\\")
    
    # Close table
    table_content.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    # Write table to file
    with open(table_output_path, 'w') as f:
        f.write('\n'.join(table_content))