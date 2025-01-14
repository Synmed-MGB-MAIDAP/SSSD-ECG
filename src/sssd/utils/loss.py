from wavetools.metrics.spectral import MelSpectrogramLoss
import torch

# define loss function
mel_loss = MelSpectrogramLoss(
        window_lengths=[512],
        n_mels=[64, 128],
        mel_fmin=[0, 0],
        mel_fmax=[200, 200],
        loss_fn=torch.nn.L1Loss(),
        clamp_eps=1e-5,
        log_weight=1.0,
        mag_weight=1.0,
        weight=1.0,
        pow=2,
    )