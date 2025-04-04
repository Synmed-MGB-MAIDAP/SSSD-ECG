import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk

# from sssd.utils.demographics_mapping as demographics_mapping import thresholds_15

label_order = ['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_', 'TRIGU', 'VCLVH', 'WPW']
new_label_order = ['1AVB', '2AVB', '3AVB', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'CLBBB', 'CRBBB', 'EL', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LPFB', 'LVH', 'NORM', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'WPW', 'ABQRS', 'DIG', 'HVOLT', 'INVT', 'LNGQT', 'LOWT', 'LPR', 'LVOLT', 'NDT', 'NST_', 'NT_', 'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'STD_', 'STE_', 'TAB_', 'VCLVH', 'AFIB', 'AFLT', 'BIGU', 'PACE', 'PSVT', 'SARRH', 'SBRAD', 'SR', 'STACH', 'SVARR', 'SVTAC', 'TRIGU']
labels_15 = [["NORM"], ["SR"], ["SARRH"], ["STACH"], ["SBRAD"], ["PVC"], ["PAC"], ["AFLT"], ["AFIB"], ['1AVB'], ['CLBBB', 'ILBBB'], ['CRBBB', 'IRBBB'], ['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'LMI', 'IPLMI', 'IPMI', 'PMI'], ['NDT', 'NST_', 'DIG', 'LNGQT', 'ISC_', 'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA', 'ANEUR', 'EL', 'ISCAN'], ['PACE']]
label_15 = ['normal', 'sn', 'sna', 'snt', 'snb', 'pvc', 'pac', 'afl', 'af', 'avbi', 'lbbb', 'rbbb', 'mi', 'st', 'pacing']

diffusets_label_transform = {
        'PAC': 'pac',
        'PVC': 'pvc',
        'CLBBB': 'lbbb',
        'ILBBB': 'lbbb',
        'CRBBB': 'rbbb',
        'IRBBB': 'rbbb',
        'SR': 'sn',
        'STACH': 'snt',
        'SBRAD': 'snb',
        'SARRH': 'sna',
        'AFLT': 'afl',
        'AFIB': 'af',
        'PACE': 'pacing',
        'IMI': 'mi',
        'ASMI': 'mi',
        'ILMI': 'mi',
        'AMI': 'mi',
        'ALMI': 'mi',
        'LMI': 'mi',
        'IPLMI': 'mi',
        'IPMI': 'mi',
        'PMI': 'mi',
        'NDT': 'st',
        'NST_': 'st',
        'DIG': 'st',
        'LNGQT': 'st',
        'ISC_': 'st',
        'ISCAL': 'st',
        'ISCIN': 'st',
        'ISCIL': 'st',
        'ISCAS': 'st',
        'ISCLA': 'st',
        'ANEUR': 'st',
        'EL': 'st',
        'ISCAN': 'st',
        '1AVB': 'abvi',
        '2AVB': 'abvi',
        '3AVB': 'abvi',
        'NORM': 'normal',
    }


def one_hot_to_str(label, label_order):
    label_str = []
    if len(label.shape) == 2:
        label = label[0]
    # print("len label", len(label))
    assert len(label) == len(label_order)
    for i, l in enumerate(label):
        if l == 1:
            label_str.append(label_order[i])
    # print(label_str)
    return "-".join(label_str)

def str_to_one_hot(label_str, label_order):
    label = np.zeros(len(label_order))
    label_str_list = label_str.split("-")
    for index, label_name in enumerate(label_order):
        if label_name in label_str:
            label[index] = 1
    return label

def check_label_equal(label1, label_order1, label2, label_order2):
    label1_str = one_hot_to_str(label1, label_order1)
    label2_str = one_hot_to_str(label2, label_order2)
    print(label1_str, label2_str)
    return set(label1_str.split("-")) == set(label2_str.split("-"))


def relabel_71_to_15(one_hot_labels_71, labels_15, new_label_order):
    """
    Converts 71 one-hot encoded labels into 15 one-hot encoded labels.
    
    Args:
        one_hot_labels_71 (np.ndarray): Array of shape (2000, 71), where each row represents a one-hot encoding of 71 labels.
        labels_15 (list): A list of lists mapping 15 labels to corresponding 71-labels.
        new_label_order (list): A list defining the order of the 71 labels.
        
    Returns:
        np.ndarray: A (2000, 15) one-hot encoded label array.
    """
    # print(one_hot_labels_71.shape)
    num_samples = one_hot_labels_71.shape[0]
    num_new_labels = len(labels_15)
    
    # Initialize new one-hot labels array (2000, 15)
    one_hot_labels_15 = np.zeros((num_samples, num_new_labels), dtype=int)

    # Create a mapping from old labels to their indices
    label_to_index = {label: i for i, label in enumerate(new_label_order)}

    # Iterate through each sample
    for i in range(num_samples):
        for new_label_idx, label_group in enumerate(labels_15):
            for label in label_group:
                if label in label_to_index:
                    old_index = label_to_index[label]
                    if one_hot_labels_71[i, old_index] == 1:
                        one_hot_labels_15[i, new_label_idx] = 1

    return one_hot_labels_15

# Example usage:
# Assuming `one_hot_labels_71` is your (2000, 71) numpy array
# new_one_hot_labels_15 = relabel_71_to_15(one_hot_labels_71, labels_15, new_label_order)

def plot_ecg_comparison(real_data, synth_data, label, lead_names=None):
    """
    Plots side-by-side ECG comparisons for real and synthetic data.

    Parameters:
    - real_data: np.array or list, shape (12, time_points), Real ECG signal data.
    - synth_data: np.array or list, shape (12, time_points), Synthetic ECG signal data.
    - label: str, title of the plot.
    - lead_names: list of str (optional), names of the 12 leads.
    """
    if lead_names is None:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    fig, axes = plt.subplots(12, 2, figsize=(25, 15), sharex=True, sharey=True)

    for i in range(12):
        # Plot real_data on the left
        axes[i, 0].plot(real_data[i])
        axes[i, 0].set_ylabel(lead_names[i], fontsize=10, fontweight='bold')
        axes[i, 0].set_yticks([])  # Remove y-axis ticks for clarity
        axes[i, 0].set_xticks([]) if i < 11 else axes[i, 0].set_xlabel("Time (ms)")

        # Plot synth_data on the right
        axes[i, 1].plot(synth_data[i])
        axes[i, 1].set_yticks([])  # Remove y-axis ticks
        axes[i, 1].set_xticks([]) if i < 11 else axes[i, 1].set_xlabel("Time (ms)")

    # Titles for columns
    axes[0, 0].set_title("Real Data", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Synthetic Data", fontsize=12, fontweight='bold')

    # Add main title
    plt.suptitle(label, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to fit title
    plt.show()

import numpy as np
import neurokit2 as nk
import warnings

# def calculate_heart_rate(ecg_signal, sampling_rate=100, lead_index=1):
#     """
#     Safely calculate average heart rate from a 12x1000 ECG segment.

#     Returns:
#     - average_hr: float or None
#     - rpeaks_count: int
#     """
#     # Choose lead; fallback if invalid index
#     print(f"ecg signal", ecg_signal.shape, ecg_signal)
#     try:
#         lead = ecg_signal[lead_index, :]
#     except IndexError:
#         lead = ecg_signal[0, :]

#     try:
#         # Temporarily suppress warnings
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")

#             signals, info = nk.ecg_process(lead, sampling_rate=sampling_rate)

#         rpeaks = info.get("ECG_R_Peaks", [])

#         if rpeaks is None or len(rpeaks) < 2:
#             return None, 0  # Not enough data to calculate HR

#         rr_intervals = np.diff(rpeaks) / sampling_rate  # in seconds
#         heart_rates = 60 / rr_intervals

#         # Clean heart rate array
#         heart_rates = heart_rates[~np.isnan(heart_rates)]
#         if len(heart_rates) == 0:
#             return None, 0

#         average_hr = np.mean(heart_rates)
#         return average_hr, len(rpeaks)

#     except Exception as e:
#         print(f"[ECG HR ERROR] {e}")
#         return None, 0

import numpy as np
import neurokit2 as nk
import warnings

def calculate_heart_rate(ecg_signal, sampling_rate=100, lead_priority=[1, 0, 5]):
    """
    Safely calculate average heart rate from a 12x1000 ECG segment.
    
    Parameters:
    - ecg_signal: np.ndarray, shape (12, 1000)
    - sampling_rate: int, typically 100 or 500 Hz
    - lead_priority: list of indices to try in order (fallback if a lead fails)
    
    Returns:
    - average_hr: float or None if failed
    - rpeaks_count: int
    - used_lead: int
    """
    
    for lead_idx in lead_priority:
        try:
            lead = ecg_signal[lead_idx, :]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                signals, info = nk.ecg_process(lead, sampling_rate=sampling_rate)

            rpeaks = info.get("ECG_R_Peaks", None)

            if rpeaks is None or len(rpeaks) < 2:
                continue  # Try next lead

            rr_intervals = np.diff(rpeaks) / sampling_rate
            heart_rates = 60 / rr_intervals
            heart_rates = heart_rates[~np.isnan(heart_rates) & ~np.isinf(heart_rates)]

            if len(heart_rates) == 0:
                continue

            average_hr = float(np.mean(heart_rates))
            if np.isnan(average_hr) or np.isinf(average_hr):
                continue

            return average_hr, len(rpeaks), lead_idx

        except Exception as e:
            print(f"[Lead {lead_idx} failed] {e}")
            continue

    return None, 0, None  # All leads failed


# def map_hr_one_hot(hr):
#     """
#     Map heart rate to an one-hot vector based on the provided thresholds.
#     """
#     hr_threshold = thresholds_15['hr']
#     # Create a one-hot vector
#     one_hot_vector = [0] * (len(hr_threshold) + 1)
    
#     if not isinstance(hr, int):
#         # fall in the average hr threshold box
#         # if the number of boxes is even
#         if len(hr_threshold) % 2 == 0:
#             one_hot_vector[len(hr_threshold) // 2] = 1
#         else:
#             # randomly select one of the two middle boxes
#             if random.randint(0, 1) == 0:
#                 one_hot_vector[len(hr_threshold) // 2] = 1
#             else:
#                 one_hot_vector[len(hr_threshold) // 2 + 1] = 1
#         return one_hot_vector
    
#     hr = int(hr)
#     hr_threshold = [int(x) for x in hr_threshold]
    
#     # Find the appropriate index for the one-hot vector
#     for i, threshold in enumerate(hr_threshold):
#         if hr <= threshold:
#             one_hot_vector[i] = 1
#             break
#     else:
#         one_hot_vector[-1] = 1  # Last category if above all thresholds
    
#     return one_hot_vector
