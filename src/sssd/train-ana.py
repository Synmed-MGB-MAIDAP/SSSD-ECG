import os
import argparse
import json
import numpy as np
import torch
# import ecg_plot
import warnings
from pathlib import Path
import matplotlib.pyplot as plt


# Ray Tune/Optuna imports
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams
from eval import evaluate_model
import wandb
from utils.mimic_4_preprocess import MIMIC_IV_ECG_Dataset
from utils.demographics_mapping import categorize_demographics
from transformers import get_cosine_schedule_with_warmup
# Suppress warnings
os.environ["KEOPS_VERBOSE"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message=r"\[pyKeOps\] Warning : keyword argument dtype in Genred is deprecated ; argument is ignored.")


def plot_ecg(signal, filepath):
    signal = signal.cpu().numpy()[0]
    plt.figure()
    plt.plot(signal)
    plt.savefig(filepath, format="jpeg")
    plt.close()



def train(output_directory,
          n_iters,
          data_path,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          batch_size,
          project_name,
          experiment_name,
          model_config=None,
          tune_report=False):
  
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

    ckpt_iter = -1,
    # wandb.init(project=project_name, name=experiment_name)

    label_path = os.path.join(data_path, 'labels')
    data_path = os.path.join(data_path, 'data')
    
    # generate experiment (local) path
    local_path = "{}/ch{}_T{}_betaT{}".format(experiment_name,
                                              model_config["res_channels"], 
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
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total Parameters: {total_params:,}")
    
    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(n_iters/100),        # e.g., 1000 - 100000 iterations of warmup
    #     num_training_steps=n_iters  # total training iterations
    # )

    # # load checkpoint
    # if ckpt_iter == 'max':
    #     ckpt_iter = find_max_epoch(output_directory)
    # if ckpt_iter >= 0:
    #     try:
    #         # load checkpoint file
    #         model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
    #         checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    #         # feed model dict and optimizer state
    #         net.load_state_dict(checkpoint['model_state_dict'])
    #         if 'optimizer_state_dict' in checkpoint:
    #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #         print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    #     except:
    #         ckpt_iter = -1
    #         print('No valid checkpoint model found, start training from initialization try.')
    # else:
    #     ckpt_iter = -1
    #     print('No valid checkpoint model found, start training from initialization.')
        
    
    print("net device", next(net.parameters()).device)
    # ptbxl
    if trainset_config["finetune_dataset"] == "ptbxl_all":
        data_ptbxl = np.load(os.path.join(data_path, 'ptbxl_train_data.npy'))
        labels_ptbxl = np.load(os.path.join(label_path, 'ptbxl_train_labels.npy'))   
        
        train_data = []
        for i in range(len(data_ptbxl)):
            train_data.append([data_ptbxl[i], labels_ptbxl[i]])
        
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=6, drop_last=True)

        # Load validate data
        val_data_ptbxl = np.load(os.path.join(data_path, 'ptbxl_val_data.npy'))
        val_labels_ptbxl = np.load(os.path.join(label_path, 'ptbxl_val_labels.npy'))

        val_data = []
        for i in range(len(val_data_ptbxl)):
            val_data.append([val_data_ptbxl[i], val_labels_ptbxl[i]])

        valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=6, drop_last=False)
    
    elif trainset_config["finetune_dataset"] == "mimic_iv":
        # print("Loading MIMIC-IV dataset")
        train_data = MIMIC_IV_ECG_Dataset(dataset_path=trainset_config['data_path'], usage='train', resample_length=1024, max_samples=400000)
        val_data = MIMIC_IV_ECG_Dataset(dataset_path=trainset_config['data_path'], usage='val', resample_length=1024, max_samples=40000)
        print("Train data size: ", len(train_data))
        print("Validation data size: ", len(val_data))
        train_data = categorize_demographics(train_data)
        val_data = categorize_demographics(val_data)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers = 4, shuffle=True)
        valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,num_workers = 4, shuffle=False)
    
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    

    # training
    # n_iter = ckpt_iter + 1
    n_iter =0
    best_loss = float('inf')
    
    while n_iter < n_iters + 1:
        
        for audio, label in trainloader:
            
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            # print("print out shapes", audio.shape, label.shape)
            
            # back-propagation
            optimizer.zero_grad()
            
            X = audio, label
            
            loss = training_loss_label(net, "MSE", X, diffusion_hyperparams)

            loss.backward()
            optimizer.step()
            # scheduler.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))
                # wandb.log({"iteration": n_iter, "loss": loss.item()})

                # current_lr = scheduler.get_last_lr()[0]
                # wandb.log({"iteration": n_iter, "learning_rate": current_lr})

                # --- EVALUATION STEP ---
                val_loss = evaluate_model(net, valloader, index_8, diffusion_hyperparams)
                # print(f"[VAL] iteration: {n_iter} \tval_loss: {val_loss}")
                # wandb.log({"iteration": n_iter, "val_loss": val_loss})

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    if tune_report:
                        session.report({"loss": loss.item(), "val_loss": val_loss})



            # save checkpoint
            # if n_iter > 0 and n_iter % iters_per_ckpt == 0:
            #     checkpoint_name = '{}.pkl'.format(n_iter)
            #     torch.save({'model_state_dict': net.state_dict(),
            #                 'optimizer_state_dict': optimizer.state_dict()},
            #                os.path.join(output_directory, checkpoint_name))
            #     # wandb.save('model.pth')
            #     print('model at iteration %s is saved' % n_iter)
            #     # Log the model checkpoint as an artifact to W&B
            #     checkpoint_path = os.path.join(output_directory, checkpoint_name)
            #     wandb.save(checkpoint_path)

            n_iter += 1
    return best_loss



# Ray Tune trainable
def train_tune(config):
    # Set up model config from Ray Tune config
    model_config = {
        "in_channels": 8,
        "out_channels": 8,
        "num_res_layers": config["num_res_layers"],
        "res_channels": config["res_channels"],
        "skip_channels": config["skip_channels"],
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 1000,
        "s4_d_state": 64,
        "s4_dropout": 0.0,
        "s4_bidirectional": 1,
        "s4_layernorm": 1,
        "label_embed_dims":[32, 32, 32, 32],
        "label_embed_classes":29,
        "class_split":[[0,14], [15, 20], [21, 22], [23, 28]]
    }
    # You can adjust these paths as needed

    data_path = trainset_config['data_path']
    output_directory = "/tmp/sssd-ecg-raytunev2"
    best_loss = train(
        output_directory=output_directory,
        n_iters=100,  # Keep small for tuning
        data_path=data_path,
        iters_per_ckpt=5,
        iters_per_logging=5,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        project_name="SSSD-ECG",
        experiment_name="raytune",
        model_config=model_config,
        tune_report=True
    )
    
    session.report({"loss": best_loss})

def run_hyperparam_search():
    ray.init(ignore_reinit_error=True)
    # search_space_v1 = {
    #     "num_res_layers": tune.choice([12, 24, 36]),
    #     "res_channels": tune.choice([64, 128, 256]),
    #     "skip_channels": tune.choice([32,64, 128, 256]),
    #     "learning_rate": tune.loguniform(5e-5, 5e-3),
    #     "batch_size": tune.choice([4, 8, 16,32])
    # }
    search_space = {
        "num_res_layers": tune.choice([12, 24]),
        "res_channels": tune.choice([256]),
        "skip_channels": tune.choice([ 128, 256]),
        "learning_rate": tune.choice([0.000633822,0.00075953,0.000410726]),
        "batch_size": tune.choice([4])
    }
    algo = OptunaSearch(metric="loss", mode="min")
    scheduler = ASHAScheduler(metric="loss", mode="min",max_t=30, 
                              grace_period=10, reduction_factor=2)
    analysis = tune.run(
        train_tune,
        resources_per_trial={"gpu": 1, "cpu": 4},
        config=search_space,
        num_samples=50,
        search_alg=algo,
        scheduler=scheduler,
        storage_path="/home/anamikumari/sssd-ecg-hyperparam-ana/sssd-ecg-hyperparam-mimic/v2",
        name="sssd_ecg_optuna",
        fail_fast=False
    )
    print("Best config: ", analysis.get_best_config(metric="loss", mode="min"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/SSSD_ECG-ana.json', help='JSON config file')
    parser.add_argument('--tune', action='store_true', help='Run Ray Tune hyperparameter search')
    args = parser.parse_args()
    global diffusion_hyperparams
    global trainset_config
    diffusion_config={
        "T": 200,
        "beta_0": 0.0001,
        "beta_T": 0.02
    }

    trainset_config ={
        "segment_length":1000,
        "sampling_rate": 100,
        "finetune_dataset":"mimic_iv",
        "data_path":"/home/anamikumari/backup/mmic_iv_ecg/files/mimic-iv-ecg/1.0"
    }
    
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
    if args.tune:
        run_hyperparam_search()
    else:
        print(args.config)
        with open(args.config) as f:
            config = json.load(f)
        train_config = config["train_config"]
        model_config = config["wavenet_config"]
        project_config = config["project_config"]
        train(
            output_directory=train_config["output_directory"],
            n_iters=train_config["n_iters"],
            data_path=train_config["data_path"],
            iters_per_ckpt=train_config["iters_per_ckpt"],
            iters_per_logging=train_config["iters_per_logging"],
            learning_rate=train_config["learning_rate"],
            batch_size=train_config["batch_size"],
            project_name=project_config["project_name"],
            experiment_name=project_config["experiment_name"],
            model_config=model_config
        )