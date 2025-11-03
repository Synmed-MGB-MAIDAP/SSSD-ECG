import os
import ecg_plot
import argparse
import json
import numpy as np
import torch
import wandb
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams, sampling_label, plot_ecg_comparison
from eval import evaluate_model
import wandb
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from inference import generate_four_leads
import sys
import importlib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_ecg(signal, filepath):
    signal = signal.numpy()[0]
    ecg_plot.plot(signal, sample_rate = 100)
    plt.savefig(filepath, format="jpeg")

def train(output_directory,
          ckpt_iter,
          n_iters,
          data_path,
          iters_per_ckpt,
          iters_per_logging,
          #iters_per_test,
          learning_rate,
         batch_size,
         project_name,
         experiment_name,
         use_ptbxl,
         ptbxl_data_path,
         debug=True):
  
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
    iters_per_test (int):           number of iterations to inference on test split of the dataset and get scores
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
    wandb.log({"total_params": total_params}, step=0)
    
    # define optimizer
    print(f"[INFO] Initializing Adam optimizer with learning rate: {learning_rate}")
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    if scheduler_func is not None:            
        scheduler_num_warmup_steps = int(n_iters / 100)  # warmup for 1% of total iterations
        scheduler_num_cycles = np.ceil(n_iters / 10000)  # 10k steps in each cycle
        if scheduler_func == "cosine_warmup":
            print(f"[INFO] Using cosine scheduler with {scheduler_num_warmup_steps} warmup steps.")
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_num_warmup_steps,
                num_training_steps=n_iters  # total training iterations
            )
        elif scheduler_func == "cosine_warmup_restarts":
            print(f"[INFO] Using cosine scheduler with hard restarts, {scheduler_num_warmup_steps} warmup steps, and {scheduler_num_cycles} cycles.")
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_num_warmup_steps,
                num_training_steps=n_iters,
                num_cycles=scheduler_num_cycles,
            )
    else:
        scheduler = None
        print("[INFO] No scheduler function provided, using default optimizer without scheduling.")

    # load checkpoint
    print(f"[INFO] Loading checkpoint: ckpt_iter={ckpt_iter}")
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
        print(f"[INFO] Max checkpoint found: {ckpt_iter}")
        # ckpt_iter = 100000
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
            wandb.log({"checkpoint_loaded": ckpt_iter}, step=0)
            
        except Exception as e:
            ckpt_iter = -1
            print(f'No valid checkpoint model found, start training from initialization try. Error: {e}')
            wandb.log({"checkpoint_loaded": -1}, step=0)
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')
        wandb.log({"checkpoint_loaded": -1}, step=0)
        
    
    print("net device", next(net.parameters()).device)
    print(f"[INFO] trainset_config: {trainset_config}")
    if ("ptbxl" in trainset_config["finetune_dataset"]) or ("mimic_iv" in trainset_config["finetune_dataset"]):

        # load ptbxl or mimic_iv dataset from npy files
        print(f"[INFO] Loading {trainset_config['finetune_dataset']} dataset")

        train_data_temp = np.load(os.path.join(data_path, f'{trainset_config["finetune_dataset"]}_train_data.npy'))
        train_labels = np.load(os.path.join(label_path, f'{trainset_config["finetune_dataset"]}_train_labels.npy'))
        print("Loaded training data from ", os.path.join(data_path, f'{trainset_config["finetune_dataset"]}_train_data.npy'))
        print(f"[INFO] train data shape: {train_data_temp.shape}, train labels shape: {train_labels.shape}")
        
        train_data = []
        for i in range(len(train_data_temp)):
            train_data.append([train_data_temp[i], train_labels[i]])
        print(f"[INFO] train_data loaded: {len(train_data)} samples")
        
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

        # Load validate data
        val_data_temp = np.load(os.path.join(data_path, f'{trainset_config["finetune_dataset"]}_val_data.npy'))
        val_labels = np.load(os.path.join(label_path, f'{trainset_config["finetune_dataset"]}_val_labels.npy'))
        print("Loaded validation data from ", os.path.join(data_path, f'{trainset_config["finetune_dataset"]}_val_data.npy'))
        print(f"[INFO] val data shape: {val_data_temp.shape}, val labels shape: {val_labels.shape}")

        val_data = []
        for i in range(len(val_data_temp)):
            val_data.append([val_data_temp[i], val_labels[i]])
        print(f"[INFO] val_data loaded: {len(val_data)} samples")

        valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size, drop_last=False)
    
    else:
        print(f"[ERROR] Unknown finetune_dataset: {trainset_config['finetune_dataset']}")
        raise ValueError(f"Unknown finetune_dataset: {trainset_config['finetune_dataset']}")
    
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    
    print(f"[INFO] index_8: {index_8.tolist()}, index_4: {index_4.tolist()}")
    
    print(f"[INFO] Trainloader size: {len(trainloader)}")
    print(f"[INFO] No. of epochs: {n_iters/len(trainloader)}")
    
    # Log hyperparameters (optional)
    temp_config_to_log = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "batch_size": (
            trainloader.batch_size if hasattr(trainloader, "batch_size") else "Unknown"
        ),
        "n_iters": n_iters,
        "iters_per_ckpt": iters_per_ckpt,
        # "iters_per_test": iters_per_test,
        "iters_per_logging": iters_per_logging,
        "initial_lr": optimizer.param_groups[0]['lr'],
    }
    if scheduler is not None:
        temp_config_to_log["scheduler_func"] = scheduler_func
        temp_config_to_log["scheduler_num_warmup_steps"] = scheduler_num_warmup_steps
        temp_config_to_log["scheduler_num_cycles"] = scheduler_num_cycles
        
    wandb.config = temp_config_to_log
    print(f"[INFO] wandb config: {wandb.config}")
    
    # training
    n_iter = ckpt_iter + 1
    print(f"[INFO] Starting training loop from iteration {n_iter} to {n_iters}")
    
    # Initialize tqdm progress bar
    pbar = tqdm(range(n_iter, n_iters + 1), 
                initial=n_iter - (ckpt_iter + 1), 
                total=n_iters - (ckpt_iter + 1),
                desc="Training",
                unit="iter")
    
    step = 0
    for n_iter in pbar:
        for i, (audio, label) in enumerate(trainloader):
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            
            # back-propagation
            optimizer.zero_grad()
            
            X = audio, label

            if trainset_config['loss_fn'] == 'mel_loss':
                loss, mel, mse, orig_x_signal, reconstructed_x_signal = training_loss_label(net, trainset_config['loss_fn'], X, diffusion_hyperparams)
                wandb.log({'training loss': loss.item(), 'iteration': step})
                wandb.log({'mel loss': mel.item(), 'iteration': step})
                wandb.log({'mse loss': mse.item(), 'iteration': step})
            else:
                loss = training_loss_label(net, trainset_config['loss_fn'], X, diffusion_hyperparams)
                wandb.log({'training loss': loss.item(), 'iteration': step})
            
            loss.backward()
            optimizer.step()
            # scheduler.step()
            if debug and i>10:
                break
            step += 1
        
        # Update progress bar description with current loss
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        if n_iter % iters_per_logging == 0:
        # if True:
            print("[LOGGING] iteration: {} \tloss: {}".format(n_iter, loss.item()))
            wandb.log({"iteration": n_iter, "loss": loss.item()})
            # current_lr = scheduler.get_last_lr()[0]
            # wandb.log({"learning_rate": current_lr, "iteration": n_iter})
            # --- EVALUATION STEP ---
            print("[EVAL] Evaluating model at iteration {}".format(n_iter))
            val_loss = evaluate_model(net, valloader, index_8, diffusion_hyperparams, trainset_config['loss_fn'], debug)
            print(f"[VAL] iteration: {n_iter} \tval_loss: {val_loss}")
            wandb.log({"iteration": n_iter, "val_loss": val_loss})
            # Update progress bar with validation loss
            pbar.set_postfix({'loss': f'{loss.item():.6f}', 'val_loss': f'{val_loss:.6f}'})

        if debug and n_iter % (iters_per_logging*10) == 0:
            # --- ECG PLOTTING AND LOGGING ---
            # Choose visualization data based on configuration
            print(f"\n[VIZ] viz_split_config: {viz_split_config}")
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
                    print("[VIZ] Using MIMIC-IV validation split for visualization.")
                    # Use MIMIC-IV validation split
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
                if debug and i>2:
                    break
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
                # pdb.set_trace()
                # save all the parameters
                save_dir = os.path.join(output_directory, "val_during_train_data{}".format(n_iter))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # np.save(os.path.join(save_dir, f"real_audio_{n_iter}_{i}.npy"), real_audio.cpu().numpy())
                # np.save(os.path.join(save_dir, f"real_label_{n_iter}_{i}.npy"), real_label.cpu().numpy())
                # np.save(os.path.join(save_dir, f"synth_audio_{n_iter}_{i}.npy"), synth_audio12_np)
                # torch.save({'model_state_dict': net.state_dict(),
                #         'optimizer_state_dict': optimizer.state_dict()},
                #         os.path.join(save_dir, "sssd_ecg_model.pkl"))
                # # save diffusion hyperparameters
                # torch.save(diffusion_hyperparams, os.path.join(save_dir, "diffusion_hyperparams.pt"))
                fig = plot_ecg_comparison(
                    real_audio_np[0],
                    synth_audio12_np[0],
                    label=f"iter{n_iter}_sample{i}",
                    return_fig=True
                )
                # # save the figure
                # fig.savefig(os.path.join(save_dir, f"ecg_comparison_{n_iter}_{i}.png"))
                # # Log the figure to W&B
                ecg_figs.append(wandb.Image(fig, caption=f"iter{n_iter}_sample{i}"))
            # Log all images as a list
            if trainset_config['loss_fn'] == "mel_loss":
                ecg_plot_path = f'{trainset_config["data_path"]}/ecg_plot'
                if not os.path.exists(ecg_plot_path):
                    os.makedirs(ecg_plot_path)
                original_x_path = f'{ecg_plot_path}/val_original_x_{n_iter}_{loss.item()}.jpg'
                reconstructed_x_path = f'{ecg_plot_path}/val_reconstructed_x_{n_iter}_{loss.item()}.jpg'
                plot_ecg(orig_x_signal, original_x_path)
                plot_ecg(reconstructed_x_signal, reconstructed_x_path)

                orig_x_im = plt.imread(original_x_path)
                recon_x_im = plt.imread(reconstructed_x_path)

                ecg_figs.append(wandb.Image(orig_x_im, caption=f"iter{n_iter}_origin{i}"))
                ecg_figs.append(wandb.Image(recon_x_im, caption=f"iter{n_iter}_recon{i}"))

            wandb.log({"ecg_comparisons": ecg_figs, "iteration": n_iter})

        # save checkpoint
        if n_iter > 0 and n_iter % iters_per_ckpt == 0:
            checkpoint_name = '{}.pkl'.format(n_iter)
            print(f"\n[CHECKPOINT] Saving checkpoint: {checkpoint_name}")
            torch.save({'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(output_directory, checkpoint_name))
            # Log the model checkpoint as an artifact to W&B
            checkpoint_path = os.path.join(output_directory, checkpoint_name)
            print(f"[CHECKPOINT] to checkpoint path, {checkpoint_path}")
            wandb.save(checkpoint_path)
            wandb.log({"checkpoint_saved": n_iter})
            
            # if n_iter % iters_per_test == 0:
            if False:
                # pdb.set_trace()
                print(f"[TEST] Running inference and evaluation at iteration {n_iter}")
                # Add inference and evals/sssd_eval to sys.path if not already present
                inference_path = os.path.dirname(os.path.abspath(__file__))
                eval_path = os.path.abspath(os.path.join(inference_path, '../../../evals/sssd_eval'))
                if inference_path not in sys.path:
                    sys.path.append(inference_path)
                if eval_path not in sys.path:
                    sys.path.append(eval_path)
                # Import generate from inference.py
                inference_mod = importlib.import_module('inference')
                # Prepare arguments for generate
                ckpt_path = output_directory
                test_output_dir = f"{output_directory}/inference_{n_iter}_test"
                test_ckpt_iter = n_iter
                test_data_path = os.path.dirname(os.path.dirname(data_path))  # go up to the original data_path
                test_experiment_name = experiment_name
                print(f"[TEST] generate parameters: {ckpt_path}, {test_output_dir}, {test_ckpt_iter}, {test_data_path}, {test_experiment_name}")
                # Run inference to generate test samples
                inference_mod.generate(
                    output_directory=test_output_dir,
                    num_samples=batch_size,  # or a fixed number if desired
                    ckpt_path=ckpt_path,
                    data_path=test_data_path,
                    ckpt_iter=test_ckpt_iter,
                    experiment_name=test_experiment_name,
                    inference_split="test"
                )
                # Import and run quick_eval
                try:
                    from quick_eval import load_data_chunks, main_eval
                    # Use the chunked data directory for generated data
                    generated_data_dir = os.path.join(ckpt_path, f"synth_test_data_{test_ckpt_iter}")
                    real_data_file = os.path.join(ckpt_path, f"synth_test_data_{test_ckpt_iter}", '..', 'real_data.npy')
                    all_labels_file = os.path.join(ckpt_path, f"synth_test_data_{test_ckpt_iter}", '..', 'all_labels.npy')
                    # Fallback: try to find the correct real/label files
                    if not os.path.exists(real_data_file):
                        real_data_file = os.path.join(ckpt_path, f"synth_test_data_{test_ckpt_iter}", 'real_data.npy')
                    if not os.path.exists(all_labels_file):
                        all_labels_file = os.path.join(ckpt_path, f"synth_test_data_{test_ckpt_iter}", 'all_labels.npy')
                    if os.path.exists(generated_data_dir) and os.path.exists(real_data_file) and os.path.exists(all_labels_file):
                        real_data = np.load(real_data_file)
                        all_labels = np.load(all_labels_file)
                        generated_data, generated_labels = load_data_chunks(generated_data_dir)
                        min_size = min(generated_data.shape[0], real_data.shape[0])
                        generated_data = generated_data[:min_size]
                        generated_labels = generated_labels[:min_size]
                        real_data = real_data[:min_size]
                        all_labels = all_labels[:min_size]
                        assert np.all(generated_labels == all_labels), "Labels do not match between generated and real data!"
                        print(f"[TEST] Running main_eval on {min_size} test samples...")
                        main_eval(real_data, generated_data, all_labels)
                    else:
                        print(f"[TEST] Skipping quick_eval: missing files or directories.\nGenerated: {generated_data_dir}\nReal: {real_data_file}\nLabels: {all_labels_file}")
                except Exception as e:
                    print(f"[TEST] quick_eval failed: {e}")
    
    # Close the progress bar
    pbar.close()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/home/zoeyhuang/MGB-MAIDAP/models/SSSD-ECG/src/sssd/config/SSSD-ECG_demographic_71_mel_+a.json',
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

    global model_config # removed label_embed_dim from the model_config
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
    viz_split_config = config.get('viz_split_config', {'use_ptbxl': True, "ptbxl_data_path": "/home/nutansahoo/MGB-MAIDAP/models/SSSD-ECG/src/ptb_xl/processed_ptb_xl_fs100"})  # Default to PTBXL if not specified
    print(f"[INFO] viz_split_config: {viz_split_config}")

    train(**train_config, **project_config, **viz_split_config)