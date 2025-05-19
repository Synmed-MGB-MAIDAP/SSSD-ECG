import os
import argparse
import json
import numpy as np
import torch
import wandb
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams, sampling_label, plot_ecg_comparison
from eval import evaluate_model
import wandb
from utils.mimic_4_preprocess import MIMIC_IV_ECG_Dataset
from utils.demographics_mapping import categorize_demographics
from transformers import get_cosine_schedule_with_warmup
from inference import generate_four_leads
import random
import pdb

def train(output_directory,
          ckpt_iter,
          n_iters,
          data_path,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
         batch_size,
         project_name,
         experiment_name,
         use_ptbxl,
         ptbxl_data_path):
  
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    """
    print("data_path", data_path)
    print(f"[INFO] label_path set to: {os.path.join(data_path, 'labels')}")
    print(f"[INFO] data_path set to: {os.path.join(data_path, 'data')}")
    label_path = os.path.join(data_path, 'labels')
    data_path = os.path.join(data_path, 'data')
    
    # generate experiment (local) path
    local_path = "{}/ch{}_T{}_betaT{}".format(experiment_name,
                                              model_config["res_channels"], 
                                              diffusion_config["T"], 
                                              diffusion_config["beta_T"])
    print(f"[INFO] Local experiment path: {local_path}")

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    print(f"[INFO] Output directory resolved to: {output_directory}")
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
        print(f"[INFO] Created output directory: {output_directory}")
    else:
        print(f"[INFO] Output directory already exists: {output_directory}")
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    print("[INFO] Mapping diffusion hyperparameters to GPU (except 'T'):")
    for key in diffusion_hyperparams:
        if key != "T":
            print(f"  - {key}")
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
            
    # predefine model
    print(f"[INFO] Instantiating model SSSD_ECG with config: {model_config}")
    net = SSSD_ECG(**model_config).cuda()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Parameters: {total_params:,}")
    wandb.log({"total_params": total_params})
    
    # define optimizer
    print(f"[INFO] Initializing Adam optimizer with learning rate: {learning_rate}")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(n_iters/100),        # e.g., 1000 - 100000 iterations of warmup
    #     num_training_steps=n_iters  # total training iterations
    # )

    # load checkpoint
    print(f"[INFO] Loading checkpoint: ckpt_iter={ckpt_iter}")
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
        print(f"[INFO] Max checkpoint found: {ckpt_iter}")
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            print(f"[INFO] Attempting to load checkpoint from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
            wandb.log({"checkpoint_loaded": ckpt_iter})
        except Exception as e:
            ckpt_iter = -1
            print(f'No valid checkpoint model found, start training from initialization try. Error: {e}')
            wandb.log({"checkpoint_loaded": -1})
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')
        wandb.log({"checkpoint_loaded": -1})
        
    
    print("net device", next(net.parameters()).device)
    # ptbxl
    print(f"[INFO] trainset_config: {trainset_config}")
    if trainset_config["finetune_dataset"] == "ptbxl_all":
        print("[INFO] Loading PTBXL dataset...")
        data_ptbxl = np.load(os.path.join(data_path, 'ptbxl_train_data.npy'))
        labels_ptbxl = np.load(os.path.join(label_path, 'ptbxl_train_labels.npy'))   
        print(f"[INFO] PTBXL train data shape: {data_ptbxl.shape}, labels shape: {labels_ptbxl.shape}")
        
        train_data = []
        for i in range(len(data_ptbxl)):
            train_data.append([data_ptbxl[i], labels_ptbxl[i]])
        print(f"[INFO] PTBXL train_data loaded: {len(train_data)} samples")
        
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

        # Load validate data
        val_data_ptbxl = np.load(os.path.join(data_path, 'ptbxl_val_data.npy'))
        val_labels_ptbxl = np.load(os.path.join(label_path, 'ptbxl_val_labels.npy'))
        print(f"[INFO] PTBXL val data shape: {val_data_ptbxl.shape}, labels shape: {val_labels_ptbxl.shape}")

        val_data = []
        for i in range(len(val_data_ptbxl)):
            val_data.append([val_data_ptbxl[i], val_labels_ptbxl[i]])
        print(f"[INFO] PTBXL val_data loaded: {len(val_data)} samples")

        valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=False)
    
    elif trainset_config["finetune_dataset"] == "mimic_iv":
        print("[INFO] Loading MIMIC-IV dataset")
        train_data = MIMIC_IV_ECG_Dataset(dataset_path=trainset_config['data_path'], usage='train', resample_length=1024)
        val_data = MIMIC_IV_ECG_Dataset(dataset_path=trainset_config['data_path'], usage='val', resample_length=1024, max_samples=1000)
        print("Train data size: ", len(train_data))
        print("Validation data size: ", len(val_data))
        train_data = categorize_demographics(train_data)
        val_data = categorize_demographics(val_data)
        print("[INFO] Demographics categorized for train and val data.")
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    else:
        print(f"[ERROR] Unknown finetune_dataset: {trainset_config['finetune_dataset']}")
        raise ValueError(f"Unknown finetune_dataset: {trainset_config['finetune_dataset']}")
    
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    
    print(f"[INFO] index_8: {index_8.tolist()}, index_4: {index_4.tolist()}")
    
    # Log hyperparameters (optional)
    wandb.config = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "batch_size": trainloader.batch_size if hasattr(trainloader, 'batch_size') else 'Unknown',
        "n_iters": n_iters,
        "iters_per_ckpt": iters_per_ckpt,
        "iters_per_logging": iters_per_logging,
    }
    print(f"[INFO] wandb config: {wandb.config}")
    # training
    n_iter = ckpt_iter + 1
    print(f"[INFO] Starting training loop from iteration {n_iter} to {n_iters}")
    
    while n_iter < n_iters + 1:
        
        for audio, label in trainloader:
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            # print("print out shapes", audio.shape, label.shape)
            
            # back-propagation
            optimizer.zero_grad()
            
            X = audio, label
            
            loss = training_loss_label(net, "MSE", X, diffusion_hyperparams)
            wandb.log({'training loss': loss.item(), 'iteration': n_iter})
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if n_iter % iters_per_logging == 0:
                print("[LOGGING] iteration: {} \tloss: {}".format(n_iter, loss.item()))
                wandb.log({"iteration": n_iter, "loss": loss.item()})

                # current_lr = scheduler.get_last_lr()[0]
                # wandb.log({"learning_rate": current_lr, "iteration": n_iter})
                # --- EVALUATION STEP ---
                print("[EVAL] Evaluating model at iteration {}".format(n_iter))
                val_loss = evaluate_model(net, valloader, index_8, diffusion_hyperparams)
                print(f"[VAL] iteration: {n_iter} \tval_loss: {val_loss}")
                wandb.log({"iteration": n_iter, "val_loss": val_loss})

                # --- ECG PLOTTING AND LOGGING ---
                # Choose visualization data based on configuration
                print(f"[VIZ] viz_split_config: {viz_split_config}")
                if viz_split_config['use_ptbxl']:
                    print("[VIZ] Using PTBXL validation split for visualization.")
                    # Use PTBXL validation split regardless of training dataset
                    # Always load PTBXL validation data for visualization if needed
                    if trainset_config["finetune_dataset"] != "ptbxl_all":
                        print(f"[VIZ] Loading PTBXL val data from {ptbxl_data_path}")
                        ptbxl_val_data = np.load(os.path.join(ptbxl_data_path, 'data/ptbxl_val_data.npy'))
                        ptbxl_val_labels = np.load(os.path.join(ptbxl_data_path, 'labels/ptbxl_val_labels.npy'))
                        ptbxl_val_data_list = []
                        for i in range(len(ptbxl_val_data)):
                            ptbxl_val_data_list.append([ptbxl_val_data[i], ptbxl_val_labels[i]])
                        ptbxl_valloader = torch.utils.data.DataLoader(ptbxl_val_data_list, shuffle=False, batch_size=6, drop_last=False)
                        viz_batches = list(ptbxl_valloader)
                    else:
                        viz_batches = list(valloader)
                else:
                    print("[VIZ] Using MIMIC-IV validation split for visualization.")
                    # Use MIMIC-IV validation split
                    viz_batches = list(valloader)
                
                num_samples = min(10, len(viz_batches))
                fixed_batches = viz_batches[:num_samples]
                ecg_figs = []
                for i, (real_audio, real_label) in enumerate(fixed_batches):
                    print(f"[VIZ] Generating ECG comparison for sample {i} at iteration {n_iter}")
                    # pdb.set_trace()
                    real_audio8 = torch.index_select(real_audio, 1, index_8).float().cuda()
                    real_label = real_label.float().cuda()
                    # Generate synthetic ECGs with the same label
                    synth_audio = sampling_label(
                        net,
                        real_audio8.shape,
                        diffusion_hyperparams,
                        cond=real_label
                    )
                    synth_audio_np = synth_audio.detach().cpu().numpy()
                    real_audio_np = real_audio.detach().cpu().numpy()
                    # Plot comparison for the first sample in the batch
                    synth_audio12 = generate_four_leads(synth_audio)
                    synth_audio12_np = synth_audio12.detach().cpu().numpy()
                    fig = plot_ecg_comparison(
                        real_audio_np[0],
                        synth_audio12_np[0],
                        label=f"iter{n_iter}_sample{i}",
                        return_fig=True
                    )
                    ecg_figs.append(wandb.Image(fig, caption=f"iter{n_iter}_sample{i}"))
                # Log all images as a list
                wandb.log({"ecg_comparisons": ecg_figs, "iteration": n_iter})

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                print(f"[CHECKPOINT] Saving checkpoint: {checkpoint_name}")
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                # Log the model checkpoint as an artifact to W&B
                checkpoint_path = os.path.join(output_directory, checkpoint_name)
                print(f"[CHECKPOINT] to checkpoint path, {checkpoint_path}")
                wandb.save(checkpoint_path)
                wandb.log({"checkpoint_saved": n_iter})

            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG_demographic_cond_interpolate_15_onehot_mimic_hypertuned_2.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    print(f"[INFO] Loading config file: {args.config}")
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)

    print("[INFO] Config loaded:")
    print(config)
    
    train_config = config["train_config"]  # training parameters
    print(f"[INFO] train_config: {train_config}")

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset
    print(f"[INFO] trainset_config: {trainset_config}")

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters
    print(f"[INFO] diffusion_config: {diffusion_config}")

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters
    print(f"[INFO] diffusion_hyperparams keys: {list(diffusion_hyperparams.keys())}")

    global model_config
    model_config = config['wavenet_config']
    print(f"[INFO] model_config: {model_config}")

    global project_config
    project_config = config['project_config']
    print(f"[INFO] project_config: {project_config}")

    # log the config
    wandb.init(project=project_config['project_name'], name=project_config['experiment_name'])
    print("[INFO] wandb initialized with project and experiment name.")
    print("config", config)

    # Add visualization split configuration
    global viz_split_config
    viz_split_config = config.get('viz_split_config', {'use_ptbxl': True, "ptbxl_data_path": "/home/shared/ptbxl_data_sssd-ecg/condition_mimic_15"})  # Default to PTBXL if not specified
    print(f"[INFO] viz_split_config: {viz_split_config}")

    train(**train_config, **project_config, **viz_split_config)

