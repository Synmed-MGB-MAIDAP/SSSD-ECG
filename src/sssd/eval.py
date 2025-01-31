import torch
from utils.util import training_loss_label

def evaluate_model(net, valloader, index_8, diffusion_hyperparams):
    net.eval()  # set model to evaluation mode
    
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():  # no gradient calculation
        for audio, label in valloader:
            audio = torch.index_select(audio, 1, index_8).float().cuda()
            label = label.float().cuda()
            
            X = audio, label
            loss = training_loss_label(net, "MSE", X, diffusion_hyperparams)
            total_loss += loss.item()
            count += 1
    
    net.train()  # switch back to training mode
    return total_loss / max(count, 1)  # average loss
