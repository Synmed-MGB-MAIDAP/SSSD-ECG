import torch
from utils.util import training_loss_label
from warnings import warn
from tqdm.auto import tqdm

def evaluate_model(net, valloader, index_8, diffusion_hyperparams, loss_fn, debug=False):
    net.eval()  # set model to evaluation mode
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():  # no gradient calculation
        for audio, label in tqdm(valloader, desc="Evaluating"):
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            
            X = audio, label
            if loss_fn == "mel_loss":
                loss, mel, mse, orig_x_signal, reconstructed_x_signal = training_loss_label(net, loss_fn, X, diffusion_hyperparams)
            else:
                loss = training_loss_label(net, loss_fn, X, diffusion_hyperparams)
            total_loss += loss.item()
            count += 1

            if debug and count>10:
                warn("\n\n\n\t\t==============================================\nDebug mode: Breaking after 10 batches\n\n\n\t\t==============================================\n")
                break
            
    
    net.train()  # switch back to training mode
    return total_loss / max(count, 1)  # average loss
