import os
import ast
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from scipy import signal
import wfdb
from wfdb import processing
from utils.demographics_mapping import map_age, map_gender, map_heartrate, categorize_demographics
import pickle

def create_encoding_vector(input_list):
    # encoding order
    categories = ['normal', 'sn', 'sna', 'snt', 'snb', 'pvc', 'pac', 'afl', 'af', 'avbi', 'lbbb', 'rbbb', 'mi', 'st', 'pacing']

    # Initialize a zero tensor of length equal to the number of categories
    encoding_vector = torch.zeros(len(categories))
    
    # Set the positions corresponding to the input list to 1
    for item in input_list:
        if item in categories:
            index = categories.index(item)
            encoding_vector[index] = 1

    return encoding_vector


def translate_text_to_label(text):
    text = text.lower()
    label = []
    # classification logic reference to 
    # https://github.com/Raiiyf/DiffuSETS_Exp/blob/097725c015e7c1d1d8c2755d38faae930619a069/test_scripts/batch_disease_gen.py#L81

    # 'PAC' in PTB-XL, 'pac' in DiffuSET
    if 'atrial premature contraction' in text or 'pac(s)' in text: 
        label.append('pac')

    # 'PVC' in PTB-XL, 'pvc' in DiffuSET
    if 'pvc' in text or 'ventricular premature' in text or 'premature ventricular' in text:
        label.append('pvc')

    if 'left bundle branch block' in text:
        label.append('lbbb')

    if 'right bundle branch block' in text:
        label.append('rbbb')

    # 'SR' in PTB-XL, 'sn' in DiffuSET
    if 'sinus rhythm' in text:
        label.append('sn')

    # 'STACH' in PTB-XL, 'snt' in DiffuSET
    if 'sinus tachycardia' in text:
        label.append('snt')

    # 'SBRAD' in PTB-XL, 'snb' in DiffuSET
    if 'sinus bradycardia' in text:
        label.append('snb')

    # 'SARRH' in PTB-XL, 'sna' in DiffuSET
    if 'sinus arrhythmia' in text:
        label.append('sna')

    if 'atrial flutter' in text:
        label.append('afl')

    if 'atrial fibrillation' in text:
        label.append('af')

    if 'pacing' in text or 'pace' in text:
        label.append('pacing')

    if 'infarct' in text:
        label.append('mi')

    if 'st junctional' in text or ' st ' in text or 'st-' in text:
        label.append('st')

    if 'degree' in text:
        label.append('avbi')
        
    # 'NORM' in PTB-XL, 'normal' in DiffuSET
    if ('normal ecg' in text) and ('abnormal' not in text):
        label.append('normal')

    return label


class MIMIC_IV_ECG_Dataset(Dataset):

    def __init__(
        self,
        dataset_path: str = "/home/kumargirish/data/mimic_files/1.0",
        other_files_path: str = "/home/kumargirish/data/mimic_files/",
        usage: str = "all",
        num_folds: int = 20,
        test_fold: int = None,
        seed: int = 42,
        resample_length: int = 1024,
        include_text_embeddings=False,
        text_embedding_paths=None,
        max_samples: int = None,
    ):

        print("Creating MIMIC dataset.")

        self.include_text_embeddings = include_text_embeddings
        if include_text_embeddings:
            if text_embedding_paths is None:
                print("Using default embedding paths")
                text_embedding_paths = [
                    "/home/kumargirish/data/mimic_files/mimic_report_embeddings.csv",
                    "/home/kumargirish/data/mimic_files/mimic_report_embeddings_II.csv",
                ]

            temp = []
            for embd_path in text_embedding_paths:
                temp.append(pd.read_csv(embd_path))
            self.text_to_embed_mapping = pd.concat(temp, ignore_index=True)
        else:
            self.text_to_embed_mapping = None
            
        print(f"Text embedding mapping shape: {self.text_to_embed_mapping.shape}")

        self.resample_length = resample_length
        self.dataset_path = dataset_path
        self.other_files_path = other_files_path

        # Use all data
        self.record_list = pd.read_csv(
            os.path.join(self.dataset_path, "record_list.csv"), low_memory=False
        )
        # Only use data having note (FUTURE)
        # self.record_list = pd.read_csv(os.path.join(self.dataset_path, 'waveform_note_links.csv'), low_memory=False)

        self.mach_mea = pd.read_csv(
            os.path.join(self.dataset_path, "machine_measurements.csv"),
            low_memory=False,
        )
        self.sheet = pd.merge(
            self.record_list, self.mach_mea, how="inner", on=["subject_id", "study_id"]
        )

        with open(os.path.join(self.other_files_path, 'exclude_list.pkl'), 'rb') as f:
            exclude_list = pickle.load(f)

        self.sheet.drop(exclude_list, inplace=True)

        # Data Cleaning, exclude mal-formed ecg
        with open(os.path.join(self.other_files_path, "bad_data_quality_mimic_4_ecg.txt"), "r") as input_file:
            bad_quality_list = [x.strip() for x in input_file.readlines()]

        with open(os.path.join(self.other_files_path, "empty_signal_mimic_4.txt"), "r") as input_file:
            empty_sig_list = [x.strip() for x in input_file.readlines()]

        self.sheet = self.sheet[~self.sheet["path"].isin(bad_quality_list)]
        self.sheet = self.sheet[~self.sheet["path"].isin(empty_sig_list)]

        patient_table_path = os.path.join(self.other_files_path, "patients.csv.gz")
        self.patient_table = pd.read_csv(
            patient_table_path, index_col="subject_id", low_memory=False
        )

        self.sheet = pd.merge(
            self.sheet, self.patient_table, how="inner", on=["subject_id", "subject_id"]
        )

        print("number of folds", len(self.sheet))
        print("number of folds", num_folds)
        # 0-17 train, 18 val, 19 test
        # split train and test data
        if usage in ['train', 'val', 'test']:
            if seed is not None:
                np.random.seed(seed)
            folds = np.random.randint(0, num_folds, size=len(self.sheet), dtype=np.int8)
            self.sheet['fold'] = folds

            if test_fold is None:
                test_fold = num_folds - 1
            val_fold = test_fold - 1
            if usage == 'train':
                sheet_mask = (self.sheet['fold'] != val_fold) & (self.sheet['fold'] != test_fold)
            elif usage == 'val':
                sheet_mask = self.sheet['fold'] == val_fold
            else:
                sheet_mask = self.sheet['fold'] == test_fold
            self.sheet = self.sheet[sheet_mask]

            # Add sampling if max_samples is set
            if max_samples is not None and len(self.sheet) > max_samples:
                self.sheet = self.sheet.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    
    # Preprocessing function for waveform data
    def _waveform_preprocess(self, x: np.ndarray):
        # x: (L=5000, C=12)

        x = np.nan_to_num(x)
        # resample x to intended length
        if self.resample_length:
            # x: (L, C) -> (resample_length, C)
            # print(f"Resampling from {x.shape[0]} to {self.resample_length}")
            x = signal.resample(x, self.resample_length)

        x = torch.as_tensor(x, dtype=torch.float)
        return x

    num = ["1st", "2nd", "3rd"]

    def _prompt_propcess(self, text):
        prompt_text = ""
        c = 0
        s = ""
        for ch in text:
            if ch == "|":
                # prompt_text += 'The ' + (num[c] if c <= 2 else str(c) + 'th') + ' diagnosis is {' + s + '}. '
                if c == 0:
                    prompt_text += "Most importantly, the 1st diagnosis is {" + s + "}."
                else:
                    prompt_text += (
                        "As a supplementary condition, the "
                        + (self.num[c] if c <= 2 else str(c + 1) + "th")
                        + " diagnosis is {"
                        + s
                        + "}."
                    )
                c += 1
                s = ""
            else:
                s += ch
        if s != "":
            if c == 0:
                prompt_text += "Most importantly, the 1st diagnosis is {" + s + "}."
            else:
                prompt_text += (
                    "As a supplementary condition, the "
                    + (self.num[c] if c <= 2 else str(c + 1) + "th")
                    + " diagnosis is {"
                    + s
                    + "}."
                )
            c += 1
            s = ""
        return prompt_text

    # Preprocessing function for text label
    def _text_preprocess(self, texts: list):
        # texts: list of 18 reports, where blank is parsed as np.NaN

        text_clean = []
        # wash nan value in texts
        for x in texts:
            if isinstance(x, str):
                text_clean.append(x)

        text_clean = "|".join(text_clean)

        text_to_embed = self._prompt_propcess(text_clean)
        embedding = self.text_to_embed_mapping.loc[
            self.text_to_embed_mapping["text"] == text_to_embed, "embedding"
        ].values[0]
        embedding = ast.literal_eval(embedding)
        
        return text_clean, embedding

    def __getitem__(self, idx: int):
        item_path = os.path.join(self.dataset_path, self.sheet["path"].iloc[idx])

        sig, fields = wfdb.rdsamp(item_path)
        x = self._waveform_preprocess(sig)
        new_freq = fields["fs"] * self.resample_length / 5000.0

        texts = [self.sheet.iloc[idx][f"report_{x}"] for x in range(18)]
        text, embedding = self._text_preprocess(texts)
        label = translate_text_to_label(text)
        encoded_label = create_encoding_vector(label)

        rr_interval = self.sheet.iloc[idx]['rr_interval'] / 1000.0

        # abnormal rr interval manually calculate 
        if rr_interval < 0.3 or rr_interval > 1.5:
            heart_rate = None
            for lead in range(12):
                # TODO: after resample, the frequency changes, right? DiffuSETs code does not change fs below.
                xqrs = processing.XQRS(sig=sig[:, lead], fs=new_freq)
                xqrs.detect(verbose=False)
                qrs_inds = xqrs.qrs_inds
                if len(qrs_inds) > 1:
                    rr_intervals = np.diff(qrs_inds) / new_freq
                    heart_rate = 60 / np.mean(rr_intervals)
                    break
            # Abort this data in later process
            # if heart_rate is None:
            #     heart_rate = 99999
            assert heart_rate is not None
                
        else:
            heart_rate = 60.0 / rr_interval

        label_dict = {
            "text": text,
            "label": label,
            "encoded_label": encoded_label,
            "subject_id": self.sheet.iloc[idx]["subject_id"],
            "hr": heart_rate,
            "age": self.sheet.iloc[idx]["anchor_age"],
            "gender": self.sheet.iloc[idx]["gender"],
            # 'note_id': self.sheet.iloc[idx]['note_id'],
        }
        
        if self.include_text_embeddings:
            label_dict["text_embedding"] = torch.tensor(embedding)
        
        # if self.include_text_embeddings:
        #     final_label = torch.cat((encoded_label, torch.tensor(embedding)), 0)
        #     return x.transpose(0, 1), final_label
        # else:
        #     return x, label_dict
        return x, label_dict

    def __len__(self) -> int:
        return len(self.sheet)


if __name__ == '__main__':
    # Original dataset

    data = MIMIC_IV_ECG_Dataset(resample_length=1000, include_text_embeddings=True)

    # train_data = MIMIC_IV_ECG_Dataset(usage='train', resample_length=1024, max_samples=100)
    # val_data = MIMIC_IV_ECG_Dataset(usage='val', resample_length=1024, max_samples=100)
    
    new_data = categorize_demographics(
        data,
        include_demographics=False,
        include_text_embedding=True,
        )
    print(new_data[0])
    
    dataloader = DataLoader(new_data, batch_size=2, shuffle=True)
    print(len(dataloader))

    for sample in dataloader:
        x, label_vec = sample
        print(x.shape, label_vec.shape)
        print(sample)
        break