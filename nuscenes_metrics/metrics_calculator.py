# Imports
import os
import time
import argparse
import numpy as np

from typing import Tuple

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricDataList, DetectionMetrics

cfg = config_factory("detection_cvpr_2019")

def evaluate(gt_boxes, pred_boxes) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
    """
    Performs the actual evaluation.
    :return: A tuple of high-level and the raw metric data.
    """
    start_time = time.time()

    # -----------------------------------
    # Step 1: Accumulate metric data for all classes and distance thresholds.
    # -----------------------------------

    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)

    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        # Compute APs.
        for dist_th in cfg.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)

        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    # Compute evaluation time.
    metrics.add_runtime(time.time() - start_time)

    return metrics, metric_data_list


# Function that creates a string with the metrics to print or write to a file later
def create_metrics_string(metrics_summary):
    metrics_str = ''
    metrics_str += f'mAP: {metrics_summary["mean_ap"]}\n'
    
    err_name_mapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    for tp_name, tp_val in metrics_summary['tp_errors'].items():
        metrics_str += f'{err_name_mapping[tp_name]}: {tp_val}\n'
    metrics_str += f'NDS: {metrics_summary["nd_score"]}\n'
    metrics_str += f'Eval time: {metrics_summary["eval_time"]}s\n\n'

    # Write per-class metrics
    metrics_str += 'Per-class results:\n'
    metrics_str += '%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\n' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE')
    class_aps = metrics_summary['mean_dist_aps']
    class_tps = metrics_summary['label_tp_errors']
    for class_name in class_aps.keys():
        metrics_str += '%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\n' % (
            class_name, 
            class_aps[class_name],
            class_tps[class_name]['trans_err'],
            class_tps[class_name]['scale_err'],
            class_tps[class_name]['orient_err'],
            class_tps[class_name]['vel_err'],
            class_tps[class_name]['attr_err'])
        
    return metrics_str


def main():
    parser = argparse.ArgumentParser(description='Evaluate metrics for a set of predictions. The script will evaluate\
                                     all the files in the folder and save the results in files with the same name as\
                                      the input files in the output_dir folder.')
    parser.add_argument('--preds_dir', type=str, help='Directory containing the predictions.')
    parser.add_argument('--gt_dir', type=str, help='Directory containing the ground truths.')
    parser.add_argument('--output_dir', type=str, help='Directory to save the metrics files.')

    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

     
    preds_list = os.listdir(args.preds_dir)

    for pred in preds_list:
        # Load predictions and ground truths
        pred_boxes, _ = load_prediction(os.path.join(args.preds_dir, pred), cfg.max_boxes_per_sample, DetectionBox, verbose=False)
        gt_boxes, _ = load_prediction(os.path.join(args.gt_dir, pred), cfg.max_boxes_per_sample, DetectionBox, verbose=False)

        # Metrics calculation
        metrics, metric_data_list = evaluate(gt_boxes, pred_boxes)
        metrics_summary = metrics.serialize()

        # Create the path for the metrics file (same name as `pred` but with a .txt extension)
        metrics_file_path = os.path.join(args.output_dir, f'{os.path.splitext(pred)[0]}.txt')

        # Open the file and write the results
        with open(metrics_file_path, 'w') as f:
            f.write(create_metrics_string(metrics_summary))

if __name__ == '__main__':
    main()
