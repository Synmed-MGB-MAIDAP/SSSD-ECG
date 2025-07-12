import os
import argparse
import json
import numpy as np
import torch
import time
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams, plot_ecg_comparison
from utils.mimic_4_preprocess import MIMIC_IV_ECG_Dataset
from utils.demographics_mapping import categorize_demographics
import matplotlib.pyplot as plt
import wandb
from warnings import warn

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


def plot_signal_pairs(real_signals, generated_signals, save_dir, chunk_idx, num_signal_pair_plots=5):
    """
    Plot pairs of real and generated signals for quality control
    """
    os.makedirs(save_dir, exist_ok=True)

    print(len(real_signals), len(generated_signals), num_signal_pair_plots)
    
    # Get the first num_samples indices
    indices = range(min(len(real_signals), num_signal_pair_plots))
    
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
    output_directory = "/home/zoeyhuang/output/test_checkpoints"
    ckpt_path = "/home/zoeyhuang/output/test_checkpoints"
    experiment_name = "SSSD_ECG_MIMIC_IV"
    inference_split = "val"
    print("num_samples: ", num_samples)
    if num_samples!=400:
        warn(f"num_samples={num_samples} is not 400, generating less data")

    # generate experiment (local) path
    local_path = "{}/ch{}_T{}_betaT{}".format(experiment_name, model_config["res_channels"], 
                                           diffusion_config["T"], 
                                           diffusion_config["beta_T"])
    local_path = experiment_name
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
    fine_tune_dataset = trainset_config["finetune_dataset"]

    if "ptbxl" in fine_tune_dataset or "mimic_iv" in fine_tune_dataset:
        
        print(f"[INFO] Loading {fine_tune_dataset} dataset")

        label_path = os.path.join(data_path, 'labels')
        data_path = os.path.join(data_path, 'data')
        
        # Load real data for comparison
        real_data = np.load(os.path.join(data_path, f'{fine_tune_dataset}_{inference_split}_data.npy'))
        labels = np.load(os.path.join(label_path, f'{fine_tune_dataset}_{inference_split}_labels.npy'))

        print("Loaded data from ", os.path.join(data_path, f'{fine_tune_dataset}_{inference_split}_data.npy'))
        print("Loaded labels from ", os.path.join(label_path, f'{fine_tune_dataset}_{inference_split}_labels.npy'))
        print("Number of samples: ", len(labels))
        print("Each label shape: ", labels[0].shape)
        
        # break down labels into chunks of 400
        chunks = []
        for i in range(0, len(labels), 400):
            if i + 400 <= len(labels):
                chunks.append(labels[i:i+400])
            else:
                chunks.append(labels[i:])
        
        signal_length = 1000 # Assuming signal length is 1000 for both datasets
    
    else:
        raise ValueError(f"Unsupported dataset: {fine_tune_dataset}. Supported datasets are 'ptbxl' and 'mimic_iv'.")
    
    print("Starting generation")
    tik = time.time()
    
    all_generated = []
    all_labels = []
    all_real_audio_used = []
    
    # Create directory for intermediate plots
    plot_dir = os.path.join(ckpt_path, f"synth_{inference_split}_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Synth data path
    synth_data_path = os.path.join(
        ckpt_path,
        f"synth_{fine_tune_dataset}_{inference_split}_data_{ckpt_iter}_samples_{num_samples}"
    )
    os.makedirs(synth_data_path, exist_ok=True)
    print("Using synth data path: ", synth_data_path)
    
    # Logging wandb config
    # Initialize wandb for visualization
    print("Initializing wandb for logging")
    wandb.init(
        project="sssd-ecg-inference", 
        name=f"{experiment_name}_inference_{ckpt_iter}",
        config = {
            "experiment_name": experiment_name,
            "ckpt_iter": ckpt_iter,
            "num_samples": num_samples,
            "data_path": data_path,
            "signal_length": signal_length,
            "inference_split": inference_split,
            "fine_tune_dataset": fine_tune_dataset
        }
    )
    
    for i, label in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        cond = torch.from_numpy(label).cuda().float()

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        m_limit = min(num_samples, len(cond))
        cond = cond[:m_limit, :]
        real_audio = real_data[i*num_samples:i*num_samples+m_limit, :]
        all_real_audio_used.append(real_audio)
        real_audio = torch.index_select(torch.from_numpy(real_audio), 1, index_8).float().cuda()
        
        print(f"Generating {m_limit} samples for chunk {i}")

        # Generate with the appropriate signal length
        generated_audio = sampling_label(net, (m_limit, 8, signal_length), 
                               diffusion_hyperparams,
                               cond=cond)
        
        # Generate 12 leads
        generated_audio12 = generate_four_leads(generated_audio)

        end.record()
        torch.cuda.synchronize()
        print(f'Generated {m_limit} samples in {int(start.elapsed_time(end)/1000)} seconds')

        # Get corresponding real data for this chunk
        start_idx = i * 400
        end_idx = min(start_idx + 400, len(real_data))
        chunk_real_data = real_data[start_idx:end_idx]
        
        # Plot intermediate results
        plot_signal_pairs(
            all_real_audio_used[-1],
            generated_audio12.detach().cpu().numpy(),
            plot_dir,
            i,
            num_signal_pair_plots=5  # Plot 5 random samples per chunk
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
    all_real_audio_used = np.concatenate(all_real_audio_used, axis=0)
    
    # Save complete results
    np.save(os.path.join(synth_data_path, 'all_samples.npy'), all_generated)
    np.save(os.path.join(synth_data_path, 'all_labels.npy'), all_labels)
    
    # Save real data for comparison
    np.save(os.path.join(synth_data_path, 'real_data.npy'), all_real_audio_used)
    
    # Plot final comparison of random samples
    final_plot_dir = os.path.join(plot_dir, 'final_comparison')
    os.makedirs(final_plot_dir, exist_ok=True)
    plot_signal_pairs(
        all_real_audio_used,
        all_generated,
        final_plot_dir,
        'final',
        num_signal_pair_plots=10  # Plot 10 random samples from the complete dataset
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
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default=10000,
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