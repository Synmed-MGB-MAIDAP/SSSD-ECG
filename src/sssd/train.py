import os
import argparse
import json
import numpy as np
import torch
import wandb
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, training_loss_label, calc_diffusion_hyperparams

import wandb

def train(output_directory,
          ckpt_iter,
          n_iters,
          data_path,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
         batch_size,
         project_name,
         experiment_name,):
  
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

    wandb.init(project=project_name, name=experiment_name)

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
    
    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')
        
        
    data_ptbxl = np.load(os.path.join(data_path, 'ptbxl_train_data.npy'))
    labels_ptbxl = np.load(os.path.join(label_path, 'ptbxl_train_labels.npy'))   
    
    train_data = []
    for i in range(len(data_ptbxl)):
        train_data.append([data_ptbxl[i], labels_ptbxl[i]])
    
        
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=6, drop_last=True)
       
    index_8 = torch.tensor([0,2,3,4,5,6,7,11])
    index_4 = torch.tensor([1,8,9,10])
    
    # Log hyperparameters (optional)
    wandb.config = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "batch_size": trainloader.batch_size if hasattr(trainloader, 'batch_size') else 'Unknown',
        "n_iters": n_iters,
        "iters_per_ckpt": iters_per_ckpt,
        "iters_per_logging": iters_per_logging,
    }
    # training
    n_iter = ckpt_iter + 1
    
    while n_iter < n_iters + 1:
        
        for audio, label in trainloader:
            
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            
            # back-propagation
            optimizer.zero_grad()
            
            X = audio, label
            
            loss = training_loss_label(net, "MSE", X, diffusion_hyperparams)
            # wandb.log({'training loss': loss.item()})
            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))
                wandb.log({"iteration": n_iter, "loss": loss.item()})


            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                # wandb.save('model.pth')
                print('model at iteration %s is saved' % n_iter)
                # Log the model checkpoint as an artifact to W&B
                checkpoint_path = os.path.join(output_directory, checkpoint_name)
                wandb.save(checkpoint_path)

            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    print(args.config)
    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)
    
    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']

    global project_config
    project_config = config['project_config']

    train(**train_config, **project_config)

