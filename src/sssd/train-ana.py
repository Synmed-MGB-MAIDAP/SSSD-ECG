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
# Suppress warnings
os.environ["KEOPS_VERBOSE"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='[pyKeOps] Warning : keyword argument dtype in Genred is deprecated ; argument is ignored.')


def plot_ecg(signal, filepath):
    signal = signal.cpu().numpy()[0]
    plt.figure()
    plt.plot(signal)
    plt.savefig(filepath, format="jpeg")
    plt.close()

def load_data(data_path, label_path):
    data_ptbxl = np.load(os.path.join(data_path, 'ptbxl_train_data.npy'))
    labels_ptbxl = np.load(os.path.join(label_path, 'ptbxl_train_labels.npy'))   
    train_data = []
    for i in range(len(data_ptbxl)):
        train_data.append([data_ptbxl[i], labels_ptbxl[i]])
    return train_data

def train(
    output_directory,
    n_iters,
    data_path,
    iters_per_ckpt,
    iters_per_logging,
    learning_rate,
    batch_size,
    project_name,
    experiment_name,
    model_config=None,
    tune_report=False
):
    # Prepare output directory
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

        # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # Model
    net = SSSD_ECG(**model_config).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Data
    label_path = os.path.join(data_path, 'labels')
    data_path = os.path.join(data_path, 'data')
    train_data = load_data(data_path, label_path)
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    # Training loop
    n_iter = 0
    best_loss = float('inf')
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])

    while n_iter < n_iters + 1:
        
        for audio, label in trainloader:
            
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            
            # back-propagation
            optimizer.zero_grad()
            
            X = audio, label
            
            loss, mel, mse,orig_x_signal,reconstructed_x_signal = training_loss_label(net, torch.nn.MSELoss(), X, diffusion_hyperparams)
            
            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    if tune_report:
                        session.report({"loss": best_loss})
                        session.report({"mse": mse})
                
                # original_x_path = Path.home() / f'MGB-MAIDAP/models/SSSD-ECG/signal_plot/original_x_{n_iter}_{loss.item()}.jpg'
                # reconstructed_x_path = Path.home() / f'MGB-MAIDAP/models/SSSD-ECG/signal_plot/recontructed_x_{n_iter}_{loss.item()}.jpg'
                # plot_ecg(orig_x_signal.detach().cpu(), original_x_path)
                # plot_ecg(reconstructed_x_signal.detach().cpu(), reconstructed_x_path)


                # orig_x_im = plt.imread(original_x_path)
                # recon_x_im = plt.imread(reconstructed_x_path)
                # wandb.log({"iteration": n_iter, "loss": loss.item(), "mel_loss": mel.item(), "mse_loss" : mse.item(),"original_signal": [wandb.Image(orig_x_im, caption=f"original_signal_{n_iter}_{loss.item()}.jpg'")],"reconstructed_signal": [wandb.Image(recon_x_im, caption=f"recontructed_signal_{n_iter}_{loss.item()}.jpg'")] })


            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                # wandb.save('model.pth')
                print('model at iteration %s is saved' % n_iter)
                # Log the model checkpoint as an artifact to W&B
                # checkpoint_path = os.path.join(output_directory, checkpoint_name)
                # wandb.save(checkpoint_path)
            # if tune_report:
            #     tune.report(loss=loss.item())
            n_iter += 1
    return best_loss


    # for epoch in range(n_iters):
    #     for audio, label in trainloader:
    #         audio = torch.tensor(audio).float().cuda()
    #         label = torch.tensor(label).float().cuda()
    #         optimizer.zero_grad()
    #         output = net(audio)
    #         loss = net.calculate_loss(output, label)
    #         loss.backward()
    #         optimizer.step()
    #         n_iter += 1

    #         if n_iter % iters_per_logging == 0:
    #             print(f"iteration: {n_iter}\tloss: {loss.item()}")
    #             if loss.item() < best_loss:
    #                 best_loss = loss.item()
    #             # Optionally plot ECG
    #             # plot_ecg(audio[0].detach().cpu(), f"original_{n_iter}.jpg")
    #             # plot_ecg(output[0].detach().cpu(), f"recon_{n_iter}.jpg")

    #         if n_iter > 0 and n_iter % iters_per_ckpt == 0:
    #             checkpoint_name = f'{n_iter}.pkl'
    #             torch.save({'model_state_dict': net.state_dict(),
    #                         'optimizer_state_dict': optimizer.state_dict()},
    #                        os.path.join(output_directory, checkpoint_name))
    #             print(f'model at iteration {n_iter} is saved')

    #         if tune_report:
    #             tune.report(loss=loss.item())

    # return best_loss

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
        "label_embed_dim": 128,
        "label_embed_classes": 71
    }
    # You can adjust these paths as needed
    
    data_path = "/home/anamikumari/backup/ptbxl_data_sssd-ecg"
    output_directory = "/tmp/sssd-ecg-raytune"
    best_loss = train(
        output_directory=output_directory,
        n_iters=500,  # Keep small for tuning
        data_path=data_path,
        iters_per_ckpt=1000,
        iters_per_logging=20,
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
    search_space = {
        "num_res_layers": tune.choice([12, 24, 36, 48]),
        "res_channels": tune.choice([32,64, 128, 256]),
        "skip_channels": tune.choice([32,64, 128, 256]),
        "learning_rate": tune.loguniform(1e-4, 1e-2,2e-4),
        "batch_size": tune.choice([4, 8, 16,32])
    }
    algo = OptunaSearch(metric="loss", mode="min")
    scheduler = ASHAScheduler(metric="loss", mode="min")
    analysis = tune.run(
        train_tune,
        resources_per_trial={"gpu": 1},
        config=search_space,
        num_samples=50,
        search_alg=algo,
        scheduler=scheduler,
        storage_path="/home/anamikumari/sssd-ecg-hyperparam-ana/sssd-ecg-hyperparam-mimic",
        name="sssd_ecg_optuna",
        fail_fast=False
    )
    print("Best config: ", analysis.get_best_config(metric="loss", mode="min"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/SSSD_ECG-ana.json', help='JSON config file')
    parser.add_argument('--tune', action='store_true', help='Run Ray Tune hyperparameter search')
    args = parser.parse_args()
    diffusion_config={
        "T": 200,
        "beta_0": 0.0001,
        "beta_T": 0.02
    }
    global diffusion_hyperparams
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