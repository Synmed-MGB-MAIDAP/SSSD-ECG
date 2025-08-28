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


def one_hot_to_str(label: np.ndarray, label_order_list: list[str]) -> str:
    """Convert a multi-hot vector into a '-' joined label string."""
    if label.ndim == 2:
        assert label.shape[0] == 1, "Expect shape (1, K) or (K,)"
        label = label[0]
    assert len(label) == len(label_order_list), "length mismatch"
    names = [label_order_list[i] for i, v in enumerate(label) if v == 1]
    return "-".join(names)

def str_to_one_hot(label_str: str, label_order_list: list[str]) -> np.ndarray:
    """Strict string-to-multi-hot conversion (fixes substring matching bug)."""
    label = np.zeros(len(label_order_list), dtype=int)
    if not label_str:
        return label
    # Split strictly by delimiter, no substring matching
    label_items = label_str.split("-")
    # Normalize to uppercase for consistency
    label_items_upper = {s.upper() for s in label_items if s}
    for idx, name in enumerate(label_order_list):
        if name.upper() in label_items_upper:
            label[idx] = 1
    return label

def check_label_equal(label1, label_order1, label2, label_order2):
    label1_str = one_hot_to_str(label1, label_order1)
    label2_str = one_hot_to_str(label2, label_order2)
    print(label1_str, label2_str)
    return set(label1_str.split("-")) == set(label2_str.split("-"))

# === 71→15 relabeling ===

def _build_group_index_map(groups: list[list[str]], full_order: list[str]) -> list[list[int]]:
    """Build index map of each group in the full order list for efficient folding."""
    pos = {name: i for i, name in enumerate(full_order)}
    idx_map = []
    for group in groups:
        idxs = []
        for gname in group:
            if gname not in pos:
                raise ValueError(f"Label '{gname}' not found in provided order.")
            idxs.append(pos[gname])
        idx_map.append(idxs)
    return idx_map

# Precompute: indices of 15 groups in new_label_order
GROUP_INDEXES_IN_71 = _build_group_index_map(labels_15, new_label_order)

def relabel_71_to_15(onehot_71: np.ndarray) -> np.ndarray:
    """
    Fold a (K=71) multi-hot vector into (15).
    Supports (71,) or (1,71) input; returns (15,).
    Rule: if any label in a group is 1, the group is 1.
    """
    if onehot_71.ndim == 2:
        assert onehot_71.shape[0] == 1 and onehot_71.shape[1] == len(new_label_order), "Expect shape (1,71)"
        vec = onehot_71[0]
    else:
        assert onehot_71.shape[0] == len(new_label_order), "Expect length 71"
        vec = onehot_71
    out = np.zeros(len(labels_15), dtype=int)
    for i, idxs in enumerate(GROUP_INDEXES_IN_71):
        if np.any(vec[idxs] == 1):
            out[i] = 1
    return out

# === End-to-end: from original 71-label list → 15 shortname list ===

def labels71_to_labels15_shortnames(raw_labels: list[str]) -> list[str]:
    """
    Input: one sample's original label list, e.g. ["SR", "PVC", "IMI"]
    Output: 15-class shortname list, e.g. ["sn", "pvc", "mi"]
    """
    # 1) Join to string for strict matching
    joined = "-".join(raw_labels)
    onehot_71 = str_to_one_hot(joined, new_label_order)
    # 2) Fold 71 → 15
    onehot_15 = relabel_71_to_15(onehot_71)
    # 3) 15 one-hot → shortname string → list
    short_str = one_hot_to_str(onehot_15, label_15)  # e.g. "sn-pvc-mi"
    return short_str.split("-") if short_str else []

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
