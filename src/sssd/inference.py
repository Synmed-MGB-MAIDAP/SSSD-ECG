import os
import argparse
import json
import pickle
import numpy as np
import torch
import random
import time
from pathlib import Path
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams

from synthwave.model.time_series.ecg_classifier import ECGClassifier
from synthwave.architecture.mets.mets_ecg_encoder import resnet18_1d
from synthwave.message.supervised_message import SupervisedMessage
from synthwave.dataset.ptb_xl.ptb_xl_dataset import PtbXlDataset

dataset = PtbXlDataset(path="/home/shared/backup/physionet.org/files/ptb-xl/1.0.3")
metadata = dataset._prepare_labels_and_metadata()
diag_superclass_mapping = dataset._diag_superclass_mapping
print ("diag_superclass_mapping", diag_superclass_mapping)
index_to_scpcode = ['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI',
       'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT',
       'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN',
       'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN',
       'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB',
       'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT',
       'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)',
       'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD',
       'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_',
       'TRIGU', 'VCLVH', 'WPW']
superclass_labels =  np.array(['CD', 'HYP', 'MI', 'NORM', 'STTC'])
youden_thresholds = {'CD': 0.772, 'HYP': 0.006, 'MI': 0.649, 'NORM': 0.029, 'STTC': 0.118}

def get_superclass_labels_from_logits(logits):
    """
    Args:
        logits: torch.Tensor of shape (batch_size, num_classes)

    Returns:
        List of lists: predicted superclass labels for each sample
    """
    probs = torch.sigmoid(logits).detach().numpy()
    thresholds = np.array([youden_thresholds[label] for label in superclass_labels])
    predicted = probs > thresholds
    indices = [np.where(row)[0] for row in predicted]
    predicted_labels = ["|".join(superclass_labels[idx].tolist()) for idx in indices]
    return predicted_labels

def generate_four_leads(tensor):
    leadI = tensor[:,0,:].unsqueeze(1)
    leadschest = tensor[:,1:7,:]
    leadavf = tensor[:,7,:].unsqueeze(1)

    leadII = (0.5*leadI) + leadavf

    leadIII = -(0.5*leadI) + leadavf
    leadavr = -(0.75*leadI) -(0.5*leadavf)
    leadavl = (0.75*leadI) - (0.5*leadavf)

    leads12 = torch.cat([leadI, leadII, leadschest, leadIII, leadavr, leadavl, leadavf], dim=1)

    return leads12


def generate(output_directory,
             data_path,
             model_path,
             classification_model_ckpt):
    
    
    """
    Generate data based on ground truth 

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded; 
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to dataset, numpy array.
    """

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config).cuda()
    print_size(net)

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(1))
    except:
        raise Exception('No valid model found')

    # load the labels which will be used for generation
    with open(os.path.join(data_path, 'superclass_label_counts.pkl'), 'rb') as f:
        result = pickle.load(f) 
    
    superclass_model_checkpoint =  torch.load(classification_model_ckpt, map_location='cuda:0', weights_only=False)
    # batch_size = 256
    num_channels = 12
    projection_size = 5
    encoder = resnet18_1d(
        in_channels=num_channels,
        projection_size=projection_size
    )
    superclass_model = ECGClassifier(
        encoder=encoder,
        device='cuda',
        device_id=0
    )
    superclass_model.load_state_dict(superclass_model_checkpoint['model'], strict=False)
    superclass_model.eval()
    superclass_model.cuda()

    all_superclass_results = []
    for class_name, label_frequency in result.items():
        if class_name == '':
            continue
        elif class_name == 'NORM':
            size=10000
        else:
            size = 1000
        arrays, counts = zip(*label_frequency)
        counts = np.array(counts)
        # probabilities = counts / counts.sum()
        indices = np.random.choice(len(arrays), size=size, replace=True)
        random_sample = np.array([arrays[i] for i in indices])

    
        # break down labels into chunks of 400
        chunks = []
        ## TRY with 4 samples at a time
        for i in range(0, len(random_sample), 4):
            if i + 4 <= len(random_sample):
                chunks.append(random_sample[i:i+4])
            else:
                chunks.append(random_sample[i:])
        
        print("Starting generation")
        tik = time.time()
        for i, label in enumerate(chunks):
            cond = torch.from_numpy(label).cuda().float()

            # inference
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            num_samples = len(cond)
            
            print("Generating {} samples for chunk {}".format(num_samples, i))

            generated_audio = sampling_label(net, (num_samples,8,1000), 
                                diffusion_hyperparams,
                                cond=cond)
            
            generated_audio12 = generate_four_leads(generated_audio)

            end.record()
            torch.cuda.synchronize()
            print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                                i, 
                                                                                int(start.elapsed_time(end)/1000)))

        
            # use the above generated ecg to classify the ECGs into different superclasses
            # create batch
            # shape of generated audio should be (batch_size, num_channels, 1000)
            message = SupervisedMessage(inputs=generated_audio12, targets=cond) 
            # message = model(message)
            output = superclass_model(message)
            # map predictions to superclasses
            superclasses_list = get_superclass_labels_from_logits(output.outputs.cpu())

            print ("Output:", output.outputs.cpu())
            print (output.outputs.cpu().shape)

            # map predictions to labels
            predictions = output.outputs.cpu()
            input_conditions = output.targets.cpu() 
            print ("len of superclasses_list", superclasses_list, len(superclasses_list))
            print ("Shape of input conditions", input_conditions.shape)

            # Map input_conditions to superclasses
            input_scpcodes_list = [
                [index_to_scpcode[i] for i, v in enumerate(row) if v == 1]
                for row in input_conditions
            ]
            print ("Input SCP Codes list", input_scpcodes_list)
            input_superclasses_list = [
                "|".join(set([diag_superclass_mapping[code] for code in scpcodes if code in diag_superclass_mapping]))
                for scpcodes in input_scpcodes_list
            ]

            print ("Input superclass list",input_superclasses_list, len(input_superclasses_list))
            
            chunk_result = {
                "iteration": i,
                "class_name": class_name,
                "label": label,
                "predicted logits": output.outputs.cpu(),
                "input_superclasses_list": input_superclasses_list,
                "predicted_superclasses_list": superclasses_list
            }
            all_superclass_results.append(chunk_result)

            # iterate over each element and save the generated audio and labels
            folder_counters = {}

            for idx, (pred_superclass, input_superclass) in enumerate(zip(superclasses_list, input_superclasses_list)):
                if pred_superclass == input_superclass:
                    # Create folder if it doesn't exist
                    folder_path = os.path.join(output_directory, pred_superclass)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path, exist_ok=True)
                        folder_counters[pred_superclass] = 0
                    else:
                        # Initialize or increment the counter for this folder
                        if pred_superclass not in folder_counters:
                            # Count existing .npy files to continue numbering
                            existing = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
                            folder_counters[pred_superclass] = len(existing) // 3  # 3 files per sample
                        else:
                            folder_counters[pred_superclass] += 1

                    file_idx = folder_counters[pred_superclass]

                    # Save generated_audio12, cond, and predictions for this index
                    np.save(os.path.join(folder_path, f"{file_idx}_samples.npy"), generated_audio12[idx].detach().cpu().numpy())
                    np.save(os.path.join(folder_path, f"{file_idx}_labels.npy"), cond[idx].detach().cpu().numpy())
                    np.save(os.path.join(folder_path, f"{file_idx}_predicted_labels.npy"), predictions[idx].numpy())


    

                with open(os.path.join(output_directory, "track_input_n_predicted_superclass.json"), "w") as f:
                    json.dump(all_superclass_results, f, indent=2)

    tok = time.time()
    print("Total time taken: ", tok-tik)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG_inference.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    print(config)

    gen_config = config['gen_config']

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    data_path = "/home/shared/pseudolabel_experiment/prerequisites"
    model_path = "/home/shared/backup/output_sssd-ecg/raw/ch256_T200_betaT0.02/100000_download.pkl"
    classification_model_ckpt = "/home/shared/pseudolabel_experiment/prerequisites/superclass classification model/checkpoint_best.pt"

    generate(**gen_config,
             data_path=data_path,
             model_path=model_path,
            classification_model_ckpt=classification_model_ckpt)
# python inference.py -c /home/nutansahoo/MGB-MAIDAP/models/SSSD-ECG/src/sssd/config/SSSD_ECG_inference.json