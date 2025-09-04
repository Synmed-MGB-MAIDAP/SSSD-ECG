"""
Clean ECG Data Preprocessing Script for PTB-XL Dataset
Processes ECG data with demographic and clinical labels for SSSD-ECG model
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add paths for imports
sys.path.append('../')
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from clinical_ts.label_utils import *
from sssd.utils.demographics_mapping import map_heartrate

class ECGDataPreprocessor:
    """Clean ECG data preprocessing class"""
    
    def __init__(self, threshold_version="condition_15_demographic", target_fs=100):
        self.threshold_version = threshold_version
        self.target_fs = target_fs
        self.input_size = 1000
        
        # Define paths
        self.data_folder_ptb_xl = Path("/home/shared/data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
        self.target_folder_ptb_xl = Path(f"/home/shared/zoey_data/ptbxl/{threshold_version}")
        
        # Set thresholds based on version
        self.threshold_selected = self._get_thresholds()
        
        # Initialize data stats
        self.ds_mean = np.array([-0.00184586, -0.00130277, 0.00017031, -0.00091313, 
                                -0.00148835, -0.00174687, -0.00077071, -0.00207407,  
                                0.00054329, 0.00155546, -0.00114379, -0.00035649])
        self.ds_std = np.array([0.16401004, 0.1647168, 0.23374124, 0.33767231, 
                               0.33362807, 0.30583013, 0.2731171, 0.27554379, 
                               0.17128962, 0.14030828, 0.14606956, 0.14656108])
    
    def _get_thresholds(self):
        """Get threshold configuration based on version"""
        thresholds_configs = {
            "condition_v1": {
                "age": [30, 40, 50, 60, 70, 80],
                "weight": [60, 80, 100],
                "height": [160, 170, 180],
            },
            "condition_v2": {
                "age": [42, 53, 60, 67, 75, 89],
                "weight": [57, 65, 73, 82],
                "height": [157, 163, 169, 175],
            },
            "condition_v3": {
                "age": [12, 17, 34, 54, 74],
                "weight": [50, 70, 90, 110],
                "height": [150, 159, 169, 179],
            },
            "condition_bmi": {
                "bmi": [18.5, 25, 30, 35, 40]
            },
            "condition_all": {
                "age": [12, 17, 34, 54, 74],
                "weight": [50, 70, 90, 110],
                "height": [150, 159, 169, 179],
                "bmi": [18.5, 25, 30, 35, 40]
            },
            "condition_15_demographic": {
                "15": "Yes",
                "age": [12, 17, 34, 54, 74],
                "weight": [50, 70, 90, 110],
                "height": [150, 159, 169, 179],
                "hr": [60, 70, 80, 90, 100],
            }
        }
        
        return thresholds_configs.get(self.threshold_version, thresholds_configs["condition_15_demographic"])
    
    def multihot_encode(self, x, num_classes):
        """Convert list of class indices to multi-hot encoding"""
        res = np.zeros(num_classes, dtype=np.float32)
        for y in x:
            res[y] = 1
        return res
    
    def prepare_dataset(self):
        """Prepare the PTB-XL dataset with preprocessing"""
        print(f"Preparing dataset with threshold version: {self.threshold_version}")
        
        # Prepare the dataset
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = prepare_data_ptb_xl(
            self.data_folder_ptb_xl, 
            min_cnt=0, 
            target_fs=self.target_fs, 
            channels=12, 
            channel_stoi=channel_stoi_default, 
            target_folder=self.target_folder_ptb_xl, 
            thresholds=self.threshold_selected
        )
        
        # Reformat as memmap for efficiency
        reformat_as_memmap(
            df_ptb_xl, 
            self.target_folder_ptb_xl/("memmap.npy"),
            data_folder=self.target_folder_ptb_xl,
            delete_npys=True
        )
        
        return df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl
    
    def create_labels(self, df_mapped, lbl_itos):
        """Create combined labels (clinical + demographic + heart rate)"""
        print("Creating combined labels...")
        
        # Determine label types based on threshold version
        columns_selected = self.threshold_selected.keys()
        label_selected = [f"label_{col}" for col in columns_selected] + ["label_sex"]
        
        if self.threshold_version.endswith("separate"): 
            ptb_xl_label = ["label_diag", "label_form", "label_rhythm"]
        else:
            ptb_xl_label = ["label_all"]
        
        ptb_xl_label_demographics = ptb_xl_label + label_selected
        
        # Create combined labels
        all_labels = []
        for label in ptb_xl_label_demographics:
            if label not in ["label_hr"]:
                encoded = df_mapped[label + "_numeric"].apply(
                    lambda x: self.multihot_encode(x, len(lbl_itos[label]))
                )
                print(f"{label} shape: {len(lbl_itos[label])}")
                all_labels.append(np.stack(encoded))
        
        # Stack and concatenate across columns
        df_mapped["label"] = [
            np.concatenate([all_labels[i][j] for i in range(len(all_labels))])
            for j in range(len(df_mapped))
        ]
        
        return df_mapped, ptb_xl_label_demographics
    
    def create_datasets(self, df_mapped, lbl_itos):
        """Create train/val/test datasets"""
        print("Creating train/val/test splits...")
        
        tfms_ptb_xl_cpc = ToTensor()
        max_fold_id = df_mapped.strat_fold.max()
        
        # Create data splits
        df_train = df_mapped[df_mapped.strat_fold < max_fold_id - 1]
        df_val = df_mapped[df_mapped.strat_fold == max_fold_id - 1]
        df_test = df_mapped[df_mapped.strat_fold == max_fold_id]
        
        print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")
        
        # Create datasets
        datasets = {}
        for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
            chunk_length = self.input_size if split_name == "train" else 0
            stride = self.input_size
            
            datasets[split_name] = TimeseriesDatasetCrops(
                df_split, self.input_size, num_classes=len(lbl_itos),
                data_folder=self.target_folder_ptb_xl,
                chunk_length=chunk_length, min_chunk_length=self.input_size,
                stride=stride, transforms=tfms_ptb_xl_cpc,
                annotation=False, col_lbl="label",
                memmap_filename=self.target_folder_ptb_xl/("memmap.npy")
            )
        
        return datasets
    
    def save_processed_data(self, datasets):
        """Save processed data and labels to numpy files"""
        print("Saving processed data...")
        
        # Create directories
        os.makedirs(self.target_folder_ptb_xl/"data", exist_ok=True)
        os.makedirs(self.target_folder_ptb_xl/"labels", exist_ok=True)
        
        # Process and save each split
        for split_name, dataset in datasets.items():
            data_list, label_list = [], []
            
            for i in range(len(dataset)):
                # Get ECG data
                data_list.append(dataset[i].data)
                
                # Calculate heart rate and add to labels
                hr, n_peaks, _ = calculate_heart_rate(dataset[i].data)
                hr_tensor = torch.tensor(map_heartrate(hr) if hr else 0.0)
                
                # Combine original labels with heart rate
                combined_label = torch.cat([torch.tensor(dataset[i].label), hr_tensor])
                label_list.append(combined_label)
            
            # Convert to numpy arrays
            data_array = np.array(data_list)
            label_array = np.array(label_list)
            
            # Remove specific indices if needed (for certain threshold versions)
            if self.threshold_version == "condition_all_separate":
                index_to_remove = [9, 33, 36, 38]
                label_array = np.delete(label_array, index_to_remove, axis=1)
            
            # Save to files
            np.save(self.target_folder_ptb_xl/f"data/ptbxl_{split_name}_data.npy", data_array)
            np.save(self.target_folder_ptb_xl/f"labels/ptbxl_{split_name}_labels.npy", label_array)
            
            print(f"{split_name.capitalize()} - Data shape: {data_array.shape}, Labels shape: {label_array.shape}")
    
    def process_data(self):
        """Complete data processing pipeline"""
        print("Starting ECG data preprocessing pipeline...")
        
        # Step 1: Prepare dataset
        df_ptb_xl, lbl_itos_ptb_xl, mean_ptb_xl, std_ptb_xl = self.prepare_dataset()
        
        # Step 2: Load processed dataset
        df_mapped, lbl_itos, mean, std = load_dataset(self.target_folder_ptb_xl)
        
        # Step 3: Create labels
        df_mapped, label_types = self.create_labels(df_mapped, lbl_itos)
        
        # Step 4: Create datasets
        datasets = self.create_datasets(df_mapped, lbl_itos)
        
        # Step 5: Save processed data
        self.save_processed_data(datasets)
        
        print("Data preprocessing completed!")
        return df_mapped, lbl_itos, label_types

def main():
    """Main processing function"""
    preprocessor = ECGDataPreprocessor(threshold_version="condition_15_demographic")
    df_mapped, lbl_itos, label_types = preprocessor.process_data()
    
    return df_mapped, lbl_itos, label_types

if __name__ == "__main__":
    main()