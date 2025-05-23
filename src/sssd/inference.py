import os
import argparse
import json
import numpy as np
import torch
import random
import time
from pathlib import Path
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams, plot_ecg_comparison
from utils.mimic_4_preprocess import MIMIC_IV_ECG_Dataset
from utils.demographics_mapping import categorize_demographics
import matplotlib.pyplot as plt
import wandb
import pdb


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


def plot_signal_pairs(real_signals, generated_signals, save_dir, chunk_idx, num_samples=5):
    """
    Plot pairs of real and generated signals for quality control
    """
    os.makedirs(save_dir, exist_ok=True)

    print(len(real_signals), len(generated_signals), num_samples)
    
    # Randomly select samples if we have more than num_samples
    if len(real_signals) > num_samples:
        indices = np.random.choice(len(real_signals), num_samples, replace=False)
        # indicies equal to num_samples
        indices = np.arange(num_samples)
    else:
        indices = range(len(real_signals))
    
    for idx in indices:
        real = real_signals[idx]
        gen = generated_signals[idx]
        
        # Plot 8-lead comparison
        fig = plot_ecg_comparison(
            real,
            gen,
            label=f"chunk{chunk_idx}_sample{idx}",
            return_fig=True
        )
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, f'chunk{chunk_idx}_sample{idx}_8lead.png'))
        plt.close(fig)
        
        # Plot 12-lead comparison
        fig, axes = plt.subplots(12, 1, figsize=(15, 24), sharex=True)
        for lead in range(12):
            axes[lead].plot(real[lead], label='Real', color='blue')
            axes[lead].plot(gen[lead], label='Generated', color='orange', alpha=0.7)
            axes[lead].set_ylabel(f'Lead {lead+1}')
            if lead == 0:
                axes[lead].set_title(f'Chunk {chunk_idx} - Sample {idx}')
        axes[-1].set_xlabel('Time')
        axes[0].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'chunk{chunk_idx}_sample{idx}_12lead.png'))
        plt.close(fig)


def generate(output_directory,
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter,
             experiment_name,
             inference_split="test"):
    
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
    print("num_samples: ", num_samples)

    # Initialize wandb for visualization
    wandb.init(project="sssd-ecg-inference", name=f"{experiment_name}_inference_{ckpt_iter}")

    # generate experiment (local) path
    local_path = "{}/ch{}_T{}_betaT{}".format(experiment_name, model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSD_ECG(**model_config).cuda()
    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, '{}.pkl'.format(ckpt_iter))
    try:
        print('Loading model from %s' % model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

    # Define the same lead selection as in training
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])

    # Load data based on dataset type
    if trainset_config["finetune_dataset"] == "ptbxl_all":
        print("Loading PTBXL dataset")
        label_path = os.path.join(data_path, 'labels')
        data_path = os.path.join(data_path, 'data')
        
        # Load real data for comparison
        real_data = np.load(os.path.join(data_path, f'ptbxl_{inference_split}_data.npy'))
        labels = np.load(os.path.join(label_path, f'ptbxl_{inference_split}_labels.npy'))
        
        print("Loaded data from ", os.path.join(data_path, f'ptbxl_{inference_split}_data.npy'))
        print("Loaded labels from ", os.path.join(label_path, f'ptbxl_{inference_split}_labels.npy'))
        print("Number of samples: ", len(labels))
        print("Each label shape: ", labels[0].shape)
        
        # break down labels into chunks of 400
        chunks = []
        for i in range(0, len(labels), 400):
            if i + 400 <= len(labels):
                chunks.append(labels[i:i+400])
            else:
                chunks.append(labels[i:])
        
        signal_length = 1000  # PTBXL signal length
        
    elif trainset_config["finetune_dataset"] == "mimic_iv":
        print("Loading MIMIC-IV dataset")
        
        include_text_embed = trainset_config["include_text_embeddings"] == 1
        print(f"[INFO] include_text_embeddings: {include_text_embed}")
            
        test_data = MIMIC_IV_ECG_Dataset(
            dataset_path=trainset_config['data_path'], 
            usage='train', 
            resample_length=1024,
            include_text_embeddings=include_text_embed
        )
        if not include_text_embed:
            test_data = categorize_demographics(test_data)
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
        
        print("Number of samples: ", len(labels))
        print("Each label shape: ", labels[0].shape)
        
        # break down labels into chunks of 400
        chunks = []
        for i in range(0, len(labels), 400):
            if i + 400 <= len(labels):
                chunks.append(labels[i:i+400])
            else:
                chunks.append(labels[i:])
        
        signal_length = 1024  # MIMIC-IV signal length
    
    print("Starting generation")
    tik = time.time()
    
    all_generated = []
    all_labels = []
    
    # Create directory for intermediate plots
    plot_dir = os.path.join(ckpt_path, f"synth_{inference_split}_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    for i, label in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        cond = torch.from_numpy(label).cuda().float()

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        cond = cond[:num_samples, :]
        real_audio = real_data[i*num_samples:(i+1)*num_samples, :]
        real_audio = torch.index_select(torch.from_numpy(real_audio), 1, index_8).float().cuda()
        
        print(f"Generating {num_samples} samples for chunk {i}")

        # Generate with the appropriate signal length
        generated_audio = sampling_label(net, (num_samples, 8, signal_length), 
                               diffusion_hyperparams,
                               cond=cond)
        
        # Generate 12 leads
        generated_audio12 = generate_four_leads(generated_audio)

        end.record()
        torch.cuda.synchronize()
        print(f'Generated {num_samples} samples in {int(start.elapsed_time(end)/1000)} seconds')

        # Get corresponding real data for this chunk
        start_idx = i * 400
        end_idx = min(start_idx + 400, len(real_data))
        chunk_real_data = real_data[start_idx:end_idx]
        
        # Plot intermediate results
        plot_signal_pairs(
            chunk_real_data,
            generated_audio12.detach().cpu().numpy(),
            plot_dir,
            i,
            num_samples=num_samples  # Plot 5 random samples per chunk
        )
        
        # Log some samples to wandb
        if i % 2 == 0:  # Log every other chunk to avoid too many plots
            for j in range(min(3, len(chunk_real_data))):
                fig = plot_ecg_comparison(
                    chunk_real_data[j],
                    generated_audio12[j].detach().cpu().numpy(),
                    label=f"chunk{i}_sample{j}",
                    return_fig=True
                )
                wandb.log({
                    f"chunk{i}_sample{j}": wandb.Image(fig),
                    "chunk": i,
                    "sample": j
                })
                plt.close(fig)

        # Save chunk results
        all_generated.append(generated_audio12.detach().cpu().numpy())
        all_labels.append(cond.detach().cpu().numpy())
        
        # Save intermediate results
        outfile = f'{i}_samples.npy'
        synth_data_path = os.path.join(ckpt_path, f"synth_{inference_split}_data_55000")
        if not os.path.exists(synth_data_path):
            os.makedirs(synth_data_path)
        new_out = os.path.join(synth_data_path, outfile)
        np.save(new_out, generated_audio12.detach().cpu().numpy())
        
        outfile = f'{i}_labels.npy'
        new_out = os.path.join(synth_data_path, outfile)
        np.save(new_out, cond.detach().cpu().numpy())

    # Combine all chunks
    all_generated = np.concatenate(all_generated, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Save complete results
    synth_data_path = os.path.join(ckpt_path, f"synth_{inference_split}_data_30000")
    np.save(os.path.join(synth_data_path, 'all_samples.npy'), all_generated)
    np.save(os.path.join(synth_data_path, 'all_labels.npy'), all_labels)
    
    # Save real data for comparison
    np.save(os.path.join(synth_data_path, 'real_data.npy'), real_data)
    
    # Plot final comparison of random samples
    final_plot_dir = os.path.join(plot_dir, 'final_comparison')
    os.makedirs(final_plot_dir, exist_ok=True)
    plot_signal_pairs(
        real_data,
        all_generated,
        final_plot_dir,
        'final',
        num_samples=num_samples  # Plot 10 random samples from the complete dataset
    )
    
    tok = time.time()
    print("Total time taken: ", tok-tik)
    print(f"Generated {len(all_generated)} samples in total")
    print(f"Results saved to {synth_data_path}")
    print(f"Plots saved to {plot_dir}")
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/zoeyhuang/MGB-MAIDAP/models/SSSD-ECG/src/sssd/config/SSSD_ECG_demographic_cond_interpolate_15_onehot_mimic_hypertuned_inf_mimic.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default=100000,
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=4,
                        help='Number of utterances to be generated')
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
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    global model_config
    model_config = config['wavenet_config']
    experiment_name = config['project_config']['experiment_name']
    
    generate(**gen_config,
            ckpt_iter=args.ckpt_iter,
            num_samples=args.num_samples,
            experiment_name=experiment_name,
            data_path=trainset_config["data_path"])