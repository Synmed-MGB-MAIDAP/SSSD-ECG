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

# from mimic_4_preprocess import MIMIC_IV_ECG_Dataset
import sys
sys.path.append('/home/kumargirish/repos/MGB-MAIDAP/evals/sssd_eval')
from metrics.metrics import RMSE, CorrelationCoefficient, MSE
import argparse


model_name = "raw" #  model name

def load_real_ptbxl_data(data_dir="/home/kumargirish/data/ptbxl"):
    X_test = np.load(os.path.join(data_dir, "data", "ptbxl_test_data.npy"))
    Y_test = np.load(os.path.join(data_dir, "labels", "ptbxl_test_labels.npy"))
    return X_test, Y_test

def load_real_mimic_iv_data(data_dir="/home/kumargirish/data/mimic_files/1.0", include_text_embed=True):
    test_data = MIMIC_IV_ECG_Dataset(
        dataset_path=data_dir, 
        usage='test', 
        resample_length=1024,
        include_text_embeddings=include_text_embed
    )
    if not include_text_embed:
        # test_data = categorize_demographics(test_data)
        raise NotImplementedError("Demographics categorization not implemented yet.")
    else:
        # TODO: this should be handled better
        print("[INFO] Text embeddings included, no demographics categorization.")
    
    # Convert to numpy arrays
    real_data = []
    labels = []
    for audio, label in test_data:
        real_data.append(audio.numpy())
        labels.append(label.numpy())
    real_data = np.stack(real_data)
    labels = np.stack(labels)
    
    return real_data, labels


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
    
    parser = argparse.ArgumentParser(description="Evaluate synthetic ECG data against real data.")
    parser.add_argument("--dir_path", type=str, required=True, help="Directory path for evaluation data.")
    args = parser.parse_args()
    dir_path = args.dir_path
    
    dir_path = Path(dir_path)
           
    print("Using directory for evaluation:")
    print(dir_path)
    
    real_data_file = dir_path / "real_data_used.npy"
    label_file = dir_path / "all_labels.npy"
    
    if not os.path.exists(real_data_file):
        print("Using dir path to resolve mimic vs ptbxl data: ", dir_path.name)
        if 'mimic' in dir_path.name:
            print("Using MIMIC-IV data for evaluation.")
            X_test_real, Y_test_real = load_real_mimic_iv_data()
        elif 'ptbxl' in dir_path.name:
            print("Using PTB-XL data for evaluation.")
            X_test_real, Y_test_real = load_real_ptbxl_data()
    else:
        X_test_real= np.load(real_data_file)
        Y_test_real = np.load(label_file)
    
    if X_test_real.ndim == 4:
        X_test_real = X_test_real.reshape(X_test_real.shape[0]*X_test_real.shape[1], X_test_real.shape[2], X_test_real.shape[3])
    
    X_test = np.load(os.path.join(dir_path, "all_samples.npy"))
    Y_test = Y_test_real
    
    print("real data shape: ", X_test_real.shape, Y_test_real.shape)  
    print("synthetic data shape: ", X_test.shape, Y_test.shape)
    
    # # --- change this if needed ---
    # # --- ------------------------------
    # num_samples = 400
    # # --- ------------------------------
    # # --- ------------------------------
    
    # if num_samples!=400:
    #     print(f"sampling real data to {num_samples} samples per chunk")
    #     X_test_real_sampled = []
    #     for i in range(0, len(X_test_real), 400):
    #         X_test_real_sampled.append(X_test_real[i:i+num_samples])
    #     X_test_real_sampled = np.concatenate(X_test_real_sampled)
    #     X_test_real= X_test_real_sampled
    #     print("modified real data shape: ", X_test_real.shape, Y_test_real.shape)

    #sanity check
    assert np.all(Y_test == Y_test_real) == True
    assert np.all(X_test.shape == X_test_real.shape) == True

    all_results = {}
    all_results[model_name] = main_eval(X_test_real, X_test, Y_test_real)
    
    print(all_results)
    # with open(dir_path / 'eval_results.json', 'w') as json_file:
    #     json.dump(all_results, json_file, indent=4)
    # print(f"saved results to {dir_path / 'eval_results.json'}")