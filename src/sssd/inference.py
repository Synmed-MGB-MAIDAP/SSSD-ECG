import os
import argparse
import json
import numpy as np
import torch
import random
import time
from pathlib import Path
from models.SSSD_ECG import SSSD_ECG
from utils.util import find_max_epoch, print_size, sampling_label, calc_diffusion_hyperparams


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
             num_samples,
             ckpt_path,
             data_path,
             ckpt_iter):
    
    
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

    # generate experiment (local) path
    # experiment_name = "raw"
    local_path = "ch{}_T{}_betaT{}".format(model_config["res_channels"], 
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
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

    label_path = os.path.join(data_path, 'labels')
    labels = np.load(os.path.join(label_path, 'ptbxl_test_labels.npy'))
    
    # break down labels into chunks of 400

    chunks = []
    for i in range(0, len(labels), 400):
        if i + 400 <= len(labels):
            chunks.append(labels[i:i+400])
        else:
            chunks.append(labels[i:])
    
    print("Starting generation")
    tik = time.time()
    for i, label in enumerate(chunks):
        # if i!=len(chunks)-1:
        #     continue
        cond = torch.from_numpy(label).cuda().float()

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if num_samples != len(cond):
            num_samples = len(cond)
        
        print("Generating {} samples for chunk {}".format(num_samples, i))

        generated_audio = sampling_label(net, (num_samples,8,1000), 
                               diffusion_hyperparams,
                               cond=cond)
        
        generated_audio12 = generate_four_leads(generated_audio)

        end.record()
        torch.cuda.synchronize()
        print('generated {} utterances of random_digit at iteration {} in {} seconds'.format(num_samples,
                                                                               ckpt_iter, 
                                                                               int(start.elapsed_time(end)/1000)))

       
        outfile = f'{i}_samples.npy'
        synth_data_path = os.path.join(ckpt_path, "synth_data")
        if not os.path.exists(synth_data_path):
            os.makedirs(synth_data_path)
        new_out = os.path.join(synth_data_path, outfile)
        np.save(new_out, generated_audio12.detach().cpu().numpy())
        print('saved generated samples at iteration %s' % ckpt_iter)
        
        outfile = f'{i}_labels.npy'
        new_out = os.path.join(synth_data_path, outfile)
        np.save(new_out, cond.detach().cpu().numpy())
        print('saved generated samples at iteration %s' % ckpt_iter)

    tok = time.time()
    print("Total time taken: ", tok-tik)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSD_ECG_demographic_cond.json',
                        help='JSON file for configuration')
    parser.add_argument('-ckpt_iter', '--ckpt_iter', default=100000,
                        help='Which checkpoint to use; assign a number or "max"')
    parser.add_argument('-n', '--num_samples', type=int, default=400,
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
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    model_config = config['wavenet_config']
    
    generate(**gen_config,
                ckpt_iter=args.ckpt_iter,
                num_samples=args.num_samples,
                data_path=trainset_config["data_path"])