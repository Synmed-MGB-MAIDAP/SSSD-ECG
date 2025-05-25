import os
import numpy as np
import re
import yaml
from pathlib import Path
from collections import defaultdict
import pathlib
import json
import torch
from tqdm.auto import tqdm

import sys
sys.path.append('/home/kumargirish/repos/MGB-MAIDAP/evals/sssd_eval')
from metrics.metrics import RMSE, CorrelationCoefficient, MSE, SNR

real_data_path = "/home/shared/ptbxl_data_sssd-ecg" # replace with your real data path
synthetic_data_path = "/home/shared/output_sssd-ecg/raw/ch256_T200_betaT0.02/synth_test_data" # replace with your synthetic data path
model_name = "raw" #  model name

def load_npy_data(data_dir):
    X_test = np.load(os.path.join(data_dir, "data", "ptbxl_test_data.npy"))
    Y_test = np.load(os.path.join(data_dir, "labels", "ptbxl_test_labels.npy"))
    return X_test, Y_test

def load_data_chunks(data_dir):
    # load the synthetic data
    files = os.listdir(data_dir)
    sample_files = [f for f in files if re.match(r'\d+_samples.npy', f)]
    label_files = [f for f in files if re.match(r'\d+_labels.npy', f)]
    
    # sort the files
    sample_files.sort(key=lambda x: int(x.split('_')[0]))
    label_files.sort(key=lambda x: int(x.split('_')[0]))
    
    # load and merge
    data = np.concatenate([np.load(os.path.join(data_dir, f)) for f in sample_files])
    labels = np.concatenate([np.load(os.path.join(data_dir, f)) for f in label_files])
    return data, labels

def evaluate_model(real_data, data, labels):
    metrics = {
        'mse': MSE(),
        'rmse': RMSE(),
        'corr_coeff': CorrelationCoefficient(),
        # 'snr': SNR() Uncomment if needed
    }
    results = {}
    
    for name, metric in metrics.items():
        results[name] = {
            'aggregated': metric.compute_aggregated(real_data, data),
        }
    
    return results

def main_eval(real_data, data, labels):
    results = evaluate_model(real_data, data, labels)
    
    for name, metric_results in results.items():
        print(f"Aggregated {name.replace('_', ' ').title()} Score: {metric_results['aggregated']}")
    
    return results

if __name__ == "__main__":
    # Load real data
    # X_test_real, Y_test_real = load_npy_data(real_data_path)
    # X_test_real, Y_test_real = load_mimic_data(include_text_embed=True)
    
    # X_test, Y_test = load_data_chunks(synthetic_data_path)
    
    dir_path = "/home/kumargirish/output_sssd-ecg/mimic_cn/mimic_cn/ch256_T200_betaT0.02/synth_test_data_127000"
    dir_path = "/home/kumargirish/output_sssd-ecg/mimic_cn/mimic_cn/ch256_T200_betaT0.02/synth_ptbxl_all_test_data_145000"
    # dir_path = Path(__file__).parent
    
    X_test_real = np.load(os.path.join(dir_path, "real_data_used.npy"))
    X_test_real = X_test_real.reshape(X_test_real.shape[0]*X_test_real.shape[1], X_test_real.shape[2], X_test_real.shape[3])
    Y_test_real = np.load(os.path.join(dir_path, "all_labels.npy"))
    X_test = np.load(os.path.join(dir_path, "all_samples.npy"))
    Y_test = np.load(os.path.join(dir_path, "all_labels.npy"))
    
    print("real data shape: ", X_test_real.shape, Y_test_real.shape)  
    print("synthetic data shape: ", X_test.shape, Y_test.shape)
    
    # --- change this if needed ---
    # --- ------------------------------
    num_samples = 400
    # --- ------------------------------
    # --- ------------------------------
    
    if num_samples!=400:
        print(f"sampling real data to {num_samples} samples per chunk")
        X_test_real_sampled = []
        for i in range(0, len(X_test_real), 400):
            X_test_real_sampled.append(X_test_real[i:i+num_samples])
        X_test_real_sampled = np.concatenate(X_test_real_sampled)
        X_test_real= X_test_real_sampled
        print("modified real data shape: ", X_test_real.shape, Y_test_real.shape)

    #sanity check
    assert np.all(Y_test == Y_test_real) == True
    assert np.all(X_test.shape == X_test_real.shape) == True

    all_results = {}
    all_results[model_name] = main_eval(X_test_real, X_test, Y_test_real)
    
    with open(dir_path / 'eval_results.json', 'w') as json_file:
        json.dump(all_results, json_file, indent=4)
    print(f"saved results to {dir_path / 'eval_results.json'}")