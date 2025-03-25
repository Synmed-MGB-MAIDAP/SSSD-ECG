import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from scipy import signal
import wfdb
from wfdb import processing
from utils.demographics_mapping import map_age, map_gender, map_heartrate, categorize_demographics


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
    def __init__(self,
                 dataset_path: str, 
                 usage: str='all', 
                 num_folds: int=10, 
                 test_fold: int=None, 
                 seed: int=42, 
                 resample_length: int=1000,
                 max_samples: int = None):

        self.resample_length = resample_length
        self.dataset_path = dataset_path

        # Use all data
        self.record_list = pd.read_csv(os.path.join(self.dataset_path, 'record_list.csv'), low_memory=False)
        # Only use data having note (FUTURE)
        # self.record_list = pd.read_csv(os.path.join(self.dataset_path, 'waveform_note_links.csv'), low_memory=False)

        self.mach_mea = pd.read_csv(os.path.join(self.dataset_path, 'machine_measurements.csv'), low_memory=False)
        self.sheet = pd.merge(self.record_list, self.mach_mea, how='inner', on=['subject_id', 'study_id'])

        # Limit number of samples for quick testing
        if max_samples is not None:
            self.sheet = self.sheet.sample(frac=1, random_state=seed).head(max_samples).reset_index(drop=True)

        # Data Cleaning, exclude mal-formed ecg
        with open('/home/shared/bad_data_quality_mimic_4_ecg.txt', 'r') as input_file:
            bad_quality_list = [x.strip() for x in input_file.readlines()]

        with open('/home/shared/empty_signal_mimic_4.txt', 'r') as input_file:
            empty_sig_list = [x.strip() for x in input_file.readlines()]

        self.sheet = self.sheet[~self.sheet['path'].isin(bad_quality_list)]
        self.sheet = self.sheet[~self.sheet['path'].isin(empty_sig_list)]

        patient_table_path = '/home/shared/mimic-iv-2.2/hosp/patients.csv.gz'
        self.patient_table = pd.read_csv(patient_table_path, index_col='subject_id', low_memory=False)

        self.sheet = pd.merge(self.sheet, self.patient_table, how='inner', on=['subject_id', 'subject_id'])

        # split train and test data
        if usage in ['train', 'test']:
            if seed is not None:
                np.random.seed(seed)
            folds = np.random.randint(0, num_folds, size=len(self.sheet), dtype=np.int8)
            self.sheet['fold'] = folds

            if test_fold is None:
                test_fold = num_folds - 1
            if usage == 'train':
                sheet_mask = self.sheet['fold'] != test_fold
            else:
                sheet_mask = self.sheet['fold'] == test_fold
            self.sheet = self.sheet[sheet_mask]
    
    # Preprocessing function for waveform data
    def _waveform_preprocess(self, x: np.ndarray):
        # x: (L=5000, C=12)

        x = np.nan_to_num(x)
        # resample x to intended length
        if self.resample_length:
            # x: (L, C) -> (resample_length, C)
            x = signal.resample(x, self.resample_length)

        x = torch.as_tensor(x, dtype=torch.float)
        return x
    
    # Preprocessing function for text label
    def _text_preprocess(self, texts: list):
        # texts: list of 18 reports, where blank is parsed as np.NaN

        text_clean = []
        # wash nan value in texts
        for x in texts:
            if isinstance(x, str):
                text_clean.append(x)

        # TODO: add text embedding phase
        # a simple concat way 
        text_clean = '|'.join(text_clean)

        return text_clean

    def __getitem__(self, idx: int):
        item_path = os.path.join(self.dataset_path, self.sheet['path'].iloc[idx])

        sig, fields = wfdb.rdsamp(item_path)
        x = self._waveform_preprocess(sig)
        new_freq = fields["fs"] * self.resample_length / 5000.0

        texts = [self.sheet.iloc[idx][f'report_{x}'] for x in range(18)]
        text = self._text_preprocess(texts)
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
                'text': text, 
                'label':label,
                'encoded_label': encoded_label,
                'subject_id': self.sheet.iloc[idx]['subject_id'], 
                'hr': heart_rate, 
                'age': self.sheet.iloc[idx]['anchor_age'],
                'gender': self.sheet.iloc[idx]['gender']
                # 'note_id': self.sheet.iloc[idx]['note_id'], 
                }
        # x: (L, C)
        return x, label_dict

    def __len__(self) -> int:
        return len(self.sheet)


if __name__ == '__main__':
    # Original dataset
    dataset_path = '/home/shared/mmic_iv_ecg/files/mimic-iv-ecg/1.0'
    data = MIMIC_IV_ECG_Dataset(dataset_path=dataset_path, usage='test', resample_length=1000, max_samples=100)
    new_data = categorize_demographics(data)
    print(new_data[0])
    dataloader = DataLoader(new_data, batch_size=2, shuffle=True)
    print(len(dataloader))

    for sample in dataloader:
        x, label_vec = sample
        print(x.shape, label_vec.shape)
        print(sample)
        break