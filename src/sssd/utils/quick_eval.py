## =======================================================================
## Temporary quick eval file used by Girish
## =======================================================================

import os
import numpy as np
import re
from pathlib import Path

from mimic_4_preprocess import MIMIC_IV_ECG_Dataset

import sys
sys.path.append('/home/kumargirish/repos/MGB-MAIDAP/evals/sssd_eval')
from metrics.metrics import RMSE, CorrelationCoefficient, MSE

import argparse

def load_real_ptbxl_data(data_dir="/home/kumargirish/data/ptbxl/processed/d_tog_15_cn/"):
    X_test = np.load(os.path.join(data_dir, "data", "ptbxl_test_data.npy"))
    Y_test = np.load(os.path.join(data_dir, "labels", "ptbxl_test_labels.npy"))
    return X_test, Y_test

def load_real_mimic_iv_data(data_dir="/home/kumargirish/data/mimic/resampled_len_1000_with_text_embeddings"):
    X_test = np.load(os.path.join(data_dir, "data", "mimic_iv_test_data.npy"))
    Y_test = np.load(os.path.join(data_dir, "labels", "mimic_iv_test_labels.npy"))
    return X_test, Y_test

def load_real_data(data_type):
    if data_type == "mimic":
        print("Using MIMIC-IV data for evaluation.")
        X_test_real, Y_test_real = load_real_mimic_iv_data()
    elif data_type == "ptbxl":
        print("Using PTB-XL data for evaluation.")
        X_test_real, Y_test_real = load_real_ptbxl_data()
    return X_test_real, Y_test_real


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
    
    # for name, metric_results in results.items():
    #     print(f"Aggregated {name.replace('_', ' ').title()} Score: {metric_results['aggregated']}")
    
    return results

def eval_by_dir_and_type(dir_path:str, data_type:str):
    dir_path = Path(dir_path)
    print("Using directory for evaluation:")
    print(dir_path)
    
    real_data_file = dir_path / "real_data_used.npy"
    label_file = dir_path / "all_labels.npy"
    
    if not os.path.exists(real_data_file):
        X_test_real, Y_test_real = load_real_data(data_type)
    else:
        X_test_real= np.load(real_data_file)
        Y_test_real = np.load(label_file)
    
    if X_test_real.ndim == 4:
        X_test_real = X_test_real.reshape(X_test_real.shape[0]*X_test_real.shape[1], X_test_real.shape[2], X_test_real.shape[3])
    
    X_test = np.load(os.path.join(dir_path, "all_samples.npy"))
    Y_test = Y_test_real
    
    print("real data shape: ", X_test_real.shape, Y_test_real.shape)  
    print("synthetic data shape: ", X_test.shape, Y_test.shape)

    #sanity check
    assert np.all(Y_test == Y_test_real) == True
    assert np.all(X_test.shape == X_test_real.shape) == True

    return main_eval(X_test_real, X_test, Y_test_real)


def multi_eval_with_dir_paths(dir_paths:list[str], data_type:str):
    results = []
    
    X_test_real, Y_test_real = load_real_data(data_type)
    print("real data shape: ", X_test_real.shape, Y_test_real.shape)  
    
    for dir_path in dir_paths:
        dir_path = Path(dir_path)
        
        try:
            if X_test_real.ndim == 4:
                X_test_real = X_test_real.reshape(X_test_real.shape[0]*X_test_real.shape[1], X_test_real.shape[2], X_test_real.shape[3])
            
            X_test = np.load(os.path.join(dir_path, "all_samples.npy"))
            Y_test = Y_test_real
            
            # print("synthetic data shape: ", X_test.shape, Y_test.shape)

            #sanity check
            assert np.all(Y_test == Y_test_real) == True
            assert np.all(X_test.shape == X_test_real.shape) == True

            results.append(main_eval(X_test_real, X_test, Y_test_real))
        
        except Exception as e:
            print(f" ======================== ERROR ======================== ")
            print(f"Error evaluating directory {dir_path}: {e}")
            results.append({})
                
    return results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate synthetic ECG data against real data.")
    parser.add_argument("--dir_path", type=str, required=True, help="Directory path for evaluation data.")
    parser.add_argument("--data_type", type=str, choices=["mimic", "ptbxl"], required=True, help="Type of data (mimic or ptbxl).")
    
    args = parser.parse_args()
    dir_path = args.dir_path
    data_type = args.data_type
    
    eval_by_dir_and_type(dir_path, data_type)
           
    