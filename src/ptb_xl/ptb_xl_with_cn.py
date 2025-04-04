import pandas as pd
from clinical_ts.timeseries_utils import *
from clinical_ts.ecg_utils import *
from label_utils import relabel_71_label_arr_to_15_label_one_hot_arr
from pathlib import Path
import numpy as np
import os
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="PTB-XL ECG Data Preprocessing")
    parser.add_argument(
        "--seperate_disease_labels",
        action="store_true",
        help="Seperate disease labels",
    )
    parser.add_argument(
        "--include_clinical_notes",
        action="store_true",
        help="Include clinical notes or not",
    )
    parser.add_argument(
        "--text_embd_size",
        type=str,
        choices=["2d", "3d", "all"],
        default="2d",
        help="Size of text embeddings: 2d, 3d, or all",
    )
    parser.add_argument(
        "--use_15_labels",
        action="store_true",
        help="Only use the 15 labels",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        default="/home/kumargirish/data/prerequisites/ptbxl_text_embed_with_reduced.csv",
        help="Path to the embedding file",
    )
    parser.add_argument(
        "--target_fs", type=int, default=100, help="Sampling rate (100 Hz or 500 Hz)"
    )
    parser.add_argument(
        "--data_folder_ptb_xl",
        type=str,
        default="/home/shared/physionet.org/files/ptb-xl/1.0.3",
        help="Path to the PTB-XL data folder",
    )
    parser.add_argument(
        "--base_target_folder_ptb_xl",
        type=str,
        default="/home/kumargirish/data/ptbxl_data_sssd-ecg",
        help="Path to the base directory of the target folder for processed data",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Datset name to be used as folder name and appended to base directory path",
    )
    parser.add_argument(
        "--cached_memmap_folder",
        type=str,
        default=None,
        # default="/home/kumargirish/data/ptbxl_data_sssd-ecg/cached",
        help="If memmap files have been created before, pass the path of the dir here.",
    )

    args = parser.parse_args()
    print("Arguments parsed as:")
    print(args)
    return args


def main_func(args):
    include_clinical_notes = args.include_clinical_notes
    seperate_disease_labels = args.seperate_disease_labels
    use_15_labels = args.use_15_labels
    embedding_path = args.embedding_path
    target_fs = args.target_fs
    data_folder_ptb_xl = Path(args.data_folder_ptb_xl)
    target_folder_ptb_xl = Path(args.base_target_folder_ptb_xl)

    if use_15_labels:
        assert not seperate_disease_labels

    if include_clinical_notes:
        text_embed_col_suffix = {"2d": "_2d", "3d": "_3d", "all": ""}[
            args.text_embd_size
        ]

    # Demographic thresholds
    # TODO: read from a config
    demogrpahic_thresholds = {}
    demographic_thresholds_present = len(demogrpahic_thresholds) > 0
    if demographic_thresholds_present:
        dem_th_key_name_str = "_".join([k for k in demogrpahic_thresholds.keys()])

    # Name of the dataset
    dataset_name = args.dataset_name
    if dataset_name is None:
        dataset_name = (
            ("d_sep" if seperate_disease_labels else "d_tog")
            + ("_15" if use_15_labels else "")
            + (f"_cn{text_embed_col_suffix}" if include_clinical_notes else "")
            + (
                f"_dem_keys_{dem_th_key_name_str}"
                if demographic_thresholds_present
                else ""
            )
        )
        print("Resolved datset name as: ", dataset_name)
    else:
        print("Using provided datset name: ", dataset_name)

    target_folder_ptb_xl = Path(os.path.join(target_folder_ptb_xl, dataset_name))
    print("Final target folder: ", target_folder_ptb_xl)

    # If using memmap files from a cached folder
    if args.cached_memmap_folder is not None:
        print("Copying memmap files from cached folder: ", args.cached_memmap_folder)
        print("To target folder: ", target_folder_ptb_xl)
        shutil.copytree(
            args.cached_memmap_folder,
            target_folder_ptb_xl,
            dirs_exist_ok=True,
        )

    # Memory mapping path
    if os.path.exists(target_folder_ptb_xl / "df_memmap.pkl"):
        print("Memory mapped file already exists. Skipping memmap prepration.")
    else:
        # Prepare the dataset
        df_ptb_xl, lbl_itos_ptb_xl, _, _ = prepare_data_ptb_xl(
            data_folder_ptb_xl,
            min_cnt=0,
            target_fs=target_fs,
            channels=12,
            channel_stoi=channel_stoi_default,
            target_folder=target_folder_ptb_xl,
            thresholds=demogrpahic_thresholds,
        )

        print("Label keys identified as:")
        print(lbl_itos_ptb_xl.keys())

        # reformat everything as memmap for efficiency
        reformat_as_memmap(
            df_ptb_xl,
            target_folder_ptb_xl / ("memmap.npy"),
            data_folder=target_folder_ptb_xl,
            delete_npys=True,
        )

        print("PTBXL data prepared with columns:")
        print(df_ptb_xl.columns)
        print("Dtypes:")
        print(df_ptb_xl.dtypes)

        # print the first dataline of the dataframe
        print("Sample first data:")
        print(df_ptb_xl.iloc[0])

    input_size = 1000  # Sample length

    chunkify_train = False
    chunk_length_train = input_size if chunkify_train else 0
    stride_train = input_size

    chunkify_valtest = False
    chunk_length_valtest = input_size if chunkify_valtest else 0
    stride_valtest = input_size

    df_mapped, lbl_itos, _, _ = load_dataset(target_folder_ptb_xl)
    print("=======================================")
    print("Data load complete with shape", df_mapped.shape)
    print("=======================================")

    print("lbl_itos", lbl_itos.keys())
    print(df_mapped.columns)

    def multihot_encode(x, num_classes):
        res = np.zeros(num_classes, dtype=np.float32)
        for y in x:
            res[y] = 1
        return res

    columns_selected = demogrpahic_thresholds.keys()
    print(columns_selected)
    label_selected = [f"label_{col}" for col in columns_selected]
    print("label_selected", label_selected)

    if seperate_disease_labels:
        ptb_xl_label = ["label_diag", "label_form", "label_rhythm"]
    else:
        if use_15_labels:
            ptb_xl_label = []
        else:
            ptb_xl_label = ["label_all"]

    ptb_xl_label_demographcis = ptb_xl_label + label_selected
    # Label all has 71 classes, label age has 6 classes, label sex has 2 classes, label height has 4 classes, label weight has 4 classes, label bmi has 6 classes

    if len(ptb_xl_label_demographcis) > 0:
        df_mapped["label"] = df_mapped.apply(
            lambda row: np.concatenate(
                [
                    multihot_encode(row[label + "_numeric"], len(lbl_itos[label]))
                    for label in ptb_xl_label_demographcis
                ]
            ),
            axis=1,
        )

        print("Sample label = ", df_mapped["label"].iloc[0])

    # Change from 71 labels to 15 labels one-hot encoded labels
    if use_15_labels:
        # Convert 71 labels to 15 labels one-hot

        print(df_mapped["label_all"].head())
        df_mapped["label_15"] = df_mapped["label_all"].apply(
            relabel_71_label_arr_to_15_label_one_hot_arr
        )
        print(df_mapped["label_15"].head())
        print("Example of 15 labels: ", df_mapped["label_15"].iloc[0])

        # Concatenate the 15 labels with the other labels
        if "label" in df_mapped.columns:
            print("Before merging the 15 labels")
            print("Length of labels = ", len(df_mapped["label"].iloc[0]))

            df_mapped["label"] = df_mapped.apply(
                lambda row: np.concatenate(
                    [
                        np.array(row["label_15"]).astype(float),
                        np.array(row["label"]).astype(float),
                    ]
                ),
                axis=1,
            )
        else:
            df_mapped["label"] = df_mapped["label_15"]
        print("After merging the 15 labels")

    print("Length of labels = ", len(df_mapped["label"].iloc[0]))

    # Include clinical notes if needed
    if include_clinical_notes:
        text_embed_col = "text_embed" + text_embed_col_suffix
        print("Resolved text_embed_col as: ", text_embed_col)
        text_embd_df = pd.read_csv(embedding_path)[["ecg_id", text_embed_col]]
        text_embd_df = text_embd_df.set_index("ecg_id")

        df_mapped = df_mapped.join(
            text_embd_df,
            how="inner",
            validate="one_to_one",
        )
        df_mapped["label"] = df_mapped.apply(
            lambda row: np.concatenate(
                [
                    np.array(row["label"]).astype(float),
                    np.array(row[text_embed_col].strip("[]").split(",")).astype(float),
                ]
            ),
            axis=1,
        )
        print("After including clinical notes.")
        print(len(df_mapped["label"].iloc[0]))

    tfms_ptb_xl_cpc = ToTensor()

    max_fold_id = df_mapped.strat_fold.max()
    print(df_mapped["strat_fold"].value_counts())

    df_train = df_mapped[df_mapped.strat_fold < max_fold_id - 1]
    df_val = df_mapped[df_mapped.strat_fold == max_fold_id - 1]
    df_test = df_mapped[df_mapped.strat_fold == max_fold_id]

    print("Train label: ", df_train["label"].iloc[0])

    # Here are the PTB-XL dataloaders

    ds_train = TimeseriesDatasetCrops(
        df_train,
        input_size,
        num_classes=len(lbl_itos),
        data_folder=target_folder_ptb_xl,
        chunk_length=chunk_length_train,
        min_chunk_length=input_size,
        stride=stride_train,
        transforms=tfms_ptb_xl_cpc,
        annotation=False,
        col_lbl="label",
        memmap_filename=target_folder_ptb_xl / ("memmap.npy"),
    )
    ds_val = TimeseriesDatasetCrops(
        df_val,
        input_size,
        num_classes=len(lbl_itos),
        data_folder=target_folder_ptb_xl,
        chunk_length=chunk_length_valtest,
        min_chunk_length=input_size,
        stride=stride_valtest,
        transforms=tfms_ptb_xl_cpc,
        annotation=False,
        col_lbl="label",
        memmap_filename=target_folder_ptb_xl / ("memmap.npy"),
    )
    ds_test = TimeseriesDatasetCrops(
        df_test,
        input_size,
        num_classes=len(lbl_itos),
        data_folder=target_folder_ptb_xl,
        chunk_length=chunk_length_valtest,
        min_chunk_length=input_size,
        stride=stride_valtest,
        transforms=tfms_ptb_xl_cpc,
        annotation=False,
        col_lbl="label",
        memmap_filename=target_folder_ptb_xl / ("memmap.npy"),
    )

    if not os.path.exists(target_folder_ptb_xl / "data"):
        print("Creating folders", target_folder_ptb_xl / "data")
        os.makedirs(target_folder_ptb_xl / "data")
    if not os.path.exists(target_folder_ptb_xl / "labels"):
        print("Creating folders", target_folder_ptb_xl / "labels")
        os.makedirs(target_folder_ptb_xl / "labels")

    def find_data_label_and_save_for_given_split(
        ds_split, split_name, is_all_sperate=False
    ):
        split_data_npy = []
        split_label_npy = []
        for i in range(len(ds_split)):
            split_data_npy.append(ds_split[i].data)
            split_label_npy.append(ds_split[i].label)
        split_data_npy = np.array(split_data_npy)
        split_label_npy = np.array(split_label_npy)

        if is_all_sperate:
            index_to_remove = [9, 33, 36, 38]
        else:
            index_to_remove = []
        split_label_npy = np.delete(split_label_npy, index_to_remove, axis=1)

        np.save(
            target_folder_ptb_xl / f"data/ptbxl_{split_name}_data.npy", split_data_npy
        )
        np.save(
            target_folder_ptb_xl / f"labels/ptbxl_{split_name}_labels.npy",
            split_label_npy,
        )

    # For each split
    for split_df_iter, split_name_iter in zip(
        [ds_train, ds_val, ds_test], ["train", "val", "test"]
    ):
        find_data_label_and_save_for_given_split(split_df_iter, split_name_iter)

    # Load and check the shape of the saved npy files
    train_data = np.load(target_folder_ptb_xl / "data/ptbxl_train_data.npy")
    train_labels = np.load(target_folder_ptb_xl / "labels/ptbxl_train_labels.npy")
    print("Train data shape: ", train_data.shape)
    print("Train label shape: ", train_labels.shape)


if __name__ == "__main__":
    args = parse_args()
    main_func(args)
