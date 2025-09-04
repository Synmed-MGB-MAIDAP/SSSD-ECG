import os
import numpy as np
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.metrics import RMSE, CorrelationCoefficient, MSE, SNR
import yaml
from pathlib import Path
from collections import defaultdict
import pathlib
import json
from utils.mimic_4_preprocess import MIMIC_IV_ECG_Dataset
from utils.demographics_mapping import categorize_demographics

real_data_path = "/home/shared/backup/mimic-iv-ecg/resampled_len_1000" # replace with your real data path
synthetic_data_path = "/home/claracao/output_sssd-ecg/condition_15_demographic_mel_loss_len1000/condition_15_demographic_mel_loss_len1000/ch256_T200_betaT0.02/synth_test_data" # replace with your synthetic data path
model_name = "raw" #  model name


def load_npy_data(data_dir):
    X_test = np.load(os.path.join(data_dir, "data", "mimic_iv_test_data.npy"))
    Y_test = np.load(os.path.join(data_dir, "labels", "mimic_iv_test_labels.npy"))
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
    X_test_real, Y_test_real = load_npy_data(real_data_path)
    print("real data shape: ", X_test_real.shape, Y_test_real.shape)
    X_test, Y_test = load_data_chunks(synthetic_data_path)
    print("synthetic data shape: ", X_test.shape, Y_test.shape)

    #sanity check
    assert np.all(Y_test == Y_test_real) == True

    all_results = {}
    all_results[model_name] = main_eval(X_test_real, X_test, Y_test_real)