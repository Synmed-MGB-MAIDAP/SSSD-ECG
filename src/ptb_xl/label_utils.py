import numpy as np

NEW_LABEL_ORDER = [
    "1AVB",
    "2AVB",
    "3AVB",
    "ALMI",
    "AMI",
    "ANEUR",
    "ASMI",
    "CLBBB",
    "CRBBB",
    "EL",
    "ILBBB",
    "ILMI",
    "IMI",
    "INJAL",
    "INJAS",
    "INJIL",
    "INJIN",
    "INJLA",
    "IPLMI",
    "IPMI",
    "IRBBB",
    "ISCAL",
    "ISCAN",
    "ISCAS",
    "ISCIL",
    "ISCIN",
    "ISCLA",
    "ISC_",
    "IVCD",
    "LAFB",
    "LAO/LAE",
    "LMI",
    "LPFB",
    "LVH",
    "NORM",
    "PMI",
    "RAO/RAE",
    "RVH",
    "SEHYP",
    "WPW",
    "ABQRS",
    "DIG",
    "HVOLT",
    "INVT",
    "LNGQT",
    "LOWT",
    "LPR",
    "LVOLT",
    "NDT",
    "NST_",
    "NT_",
    "PAC",
    "PRC(S)",
    "PVC",
    "QWAVE",
    "STD_",
    "STE_",
    "TAB_",
    "VCLVH",
    "AFIB",
    "AFLT",
    "BIGU",
    "PACE",
    "PSVT",
    "SARRH",
    "SBRAD",
    "SR",
    "STACH",
    "SVARR",
    "SVTAC",
    "TRIGU",
]

LABEL_ORDER_15 = [
    ["NORM"],
    ["SR"],
    ["SARRH"],
    ["STACH"],
    ["SBRAD"],
    ["PVC"],
    ["PAC"],
    ["AFLT"],
    ["AFIB"],
    ["1AVB"],
    ["CLBBB", "ILBBB"],
    ["CRBBB", "IRBBB"],
    ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "LMI", "IPLMI", "IPMI", "PMI"],
    [
        "NDT",
        "NST_",
        "DIG",
        "LNGQT",
        "ISC_",
        "ISCAL",
        "ISCIN",
        "ISCIL",
        "ISCAS",
        "ISCLA",
        "ANEUR",
        "EL",
        "ISCAN",
    ],
    ["PACE"],
]

LABEL_15_TO_INDEX_MAP = {
    l: i for i, label_group in enumerate(LABEL_ORDER_15) for l in label_group
}

LABEL_15_NAMES = [
    f"{','.join(x[:3])}{'...' if len(x)>3 else ''}" for x in LABEL_ORDER_15
]

LABEL_71_TO_15_MAP = {
    l: LABEL_15_NAMES[i]
    for i, label_group in enumerate(LABEL_ORDER_15)
    for l in label_group
}


def relabel_71_label_arr_to_15_label_one_hot_arr(arr_71_labels):
    """
    Converts 71 labels into 15 one-hot encoded labels.

    Args:
        arr_71_labels (np.ndarray): list of label names from the 71.

    Returns:
        np.ndarray: A (15) one-hot encoded label array.
    """
    num_new_labels = len(LABEL_ORDER_15)

    # Initialize new one-hot labels array (2000, 15)
    arr_15_labels_one_hot = np.zeros(num_new_labels, dtype=int)

    for label_name in arr_71_labels:
        idx_in_15_labels = LABEL_15_TO_INDEX_MAP.get(label_name, None)
        if idx_in_15_labels is not None:
            arr_15_labels_one_hot[idx_in_15_labels] = 1

    return arr_15_labels_one_hot


def relabel_ndarray_of_71_label_arr_to_15_label_one_hot_arr(arr_71_labels):
    """
    Converts 71 labels into 15 one-hot encoded labels.

    Args:
        arr_71_labels (np.ndarray): Array of shape (2000), where each row is a list of label names from the 71.

    Returns:
        np.ndarray: A (2000, 15) one-hot encoded label array.
    """
    num_samples = len(arr_71_labels)
    num_new_labels = len(LABEL_ORDER_15)

    # Initialize new one-hot labels array (2000, 15)
    arr_15_labels_one_hot = np.zeros((num_samples, num_new_labels), dtype=int)

    # Iterate through each sample
    for i in range(num_samples):
        for label_name in arr_71_labels[i]:
            idx_in_15_labels = LABEL_15_TO_INDEX_MAP.get(label_name, None)
            if idx_in_15_labels is not None:
                arr_15_labels_one_hot[i, idx_in_15_labels] = 1

    return arr_15_labels_one_hot
