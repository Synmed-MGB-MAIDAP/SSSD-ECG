import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.signal import spectrogram
import torch
import sys
sys.path.append('/MGB-MAIDAP/wavetools/src')
from wavetools.metrics.spectral import MultiScaleSTFTLoss, MelSpectrogramLoss, PhaseLoss
from wavetools.core import ECGSignal, BaseSignal

class RMSE:
    """
    Class to compute the Root Mean Squared Error (RMSE) between real and generated ECG signals.

    Methods
    -------
    compute(real_signal, generated_signal)
        Computes the RMSE between the real and generated signals.
    compute_per_lead(real_signals, generated_signals)
        Computes the RMSE for each lead.
    compute_aggregated(real_signals, generated_signals)
        Computes the aggregated RMSE for all leads.
    compute_per_label(real_signals, generated_signals, labels)
        Computes the RMSE for each diagnostic label.
    """

    @staticmethod
    def compute(real_signal, generated_signal):
        """
        Compute the Root Mean Squared Error (RMSE) between real and generated ECG signals.

        Parameters
        ----------
        real_signal : np.ndarray
            The real ECG signal.
        generated_signal : np.ndarray
            The generated ECG signal.

        Returns
        -------
        float
            The RMSE value.
        """
        return np.sqrt(np.mean((real_signal - generated_signal) ** 2))

    @staticmethod
    def compute_per_lead(real_signals, generated_signals):
        """
        Compute the RMSE for each lead.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        np.ndarray
            The RMSE values for each lead.
        """
        n_samples, n_leads, _ = real_signals.shape
        rmse_scores = np.zeros((n_samples, n_leads))

        for i in range(n_samples):
            for j in range(n_leads):
                rmse_scores[i, j] = RMSE.compute(real_signals[i, j], generated_signals[i, j])
        return rmse_scores
    
    @staticmethod
    def compute_per_lead_agg(real_signals, generated_signals):
        rmse_scores = RMSE.compute_per_lead(real_signals, generated_signals)
        rmse_scores = np.mean(rmse_scores, axis = 0)
        return rmse_scores

    @staticmethod
    def compute_aggregated(real_signals, generated_signals):
        """
        Compute the aggregated RMSE for all leads.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        float
            The aggregated RMSE value.
        """
        rmse_scores = RMSE.compute_per_lead_agg(real_signals, generated_signals)
        return np.mean(rmse_scores)

    @staticmethod
    def compute_per_label(real_signals, generated_signals, labels):
        """
        Compute the RMSE for each diagnostic label.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).
        labels : np.ndarray
            The diagnostic labels with shape (n_samples, n_labels).

        Returns
        -------
        dict
            A dictionary with labels as keys and RMSE values as values.
        """
        n_labels = labels.shape[1]
        label_rmse = {label: [] for label in range(n_labels)}

        for i in range(real_signals.shape[0]):
            for label in range(n_labels):
                if labels[i, label] == 1:
                    rmse_value = RMSE.compute(real_signals[i], generated_signals[i])
                    label_rmse[label].append(rmse_value)
        return label_rmse
    
    @staticmethod
    def compute_per_label_agg(real_signals, generated_signals, labels):
        label_rmse = RMSE.compute_per_label(real_signals, generated_signals, labels)
        # Compute mean RMSE for each label
        label_rmse = {label: np.mean(rmse_values) for label, rmse_values in label_rmse.items() if rmse_values}
        return label_rmse

class CorrelationCoefficient:
    """
    Class to compute the Pearson Correlation Coefficient between real and generated ECG signals.

    Methods
    -------
    compute(real_signal, generated_signal)
        Computes the Pearson Correlation Coefficient between the real and generated signals.
    compute_per_lead(real_signals, generated_signals)
        Computes the Correlation Coefficient for each lead.
    compute_aggregated(real_signals, generated_signals)
        Computes the aggregated Correlation Coefficient for all leads.
    compute_per_label(real_signals, generated_signals, labels)
        Computes the Correlation Coefficient for each diagnostic label.
    """

    @staticmethod
    def compute(real_signal, generated_signal):
        """
        Compute the Pearson Correlation Coefficient between real and generated ECG signals.

        Parameters
        ----------
        real_signal : np.ndarray
            The real ECG signal.
        generated_signal : np.ndarray
            The generated ECG signal.

        Returns
        -------
        float
            The Pearson Correlation Coefficient value.
        """
        return np.corrcoef(real_signal, generated_signal)[0, 1]

    @staticmethod
    def compute_per_lead(real_signals, generated_signals):
        """
        Compute the Correlation Coefficient for each lead.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        np.ndarray
            The Correlation Coefficient values for each lead.
        """
        n_samples, n_leads, _ = real_signals.shape
        corr_coeff_scores = np.zeros((n_samples, n_leads))

        for i in range(n_samples):
            for j in range(n_leads):
                corr_coeff_scores[i, j] = CorrelationCoefficient.compute(real_signals[i, j], generated_signals[i, j])

        return corr_coeff_scores
    
    @staticmethod
    def compute_per_lead_agg(real_signals, generated_signals):
        corr_coeff_scores = CorrelationCoefficient.compute_per_lead(real_signals, generated_signals)
        corr_coeff_scores = np.mean(np.abs(corr_coeff_scores), axis = 0)
        return corr_coeff_scores

    @staticmethod
    def compute_aggregated(real_signals, generated_signals):
        """
        Compute the aggregated Correlation Coefficient for all leads.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        float
            The aggregated Correlation Coefficient value.
        """
        corr_coeff_scores = CorrelationCoefficient.compute_per_lead_agg(real_signals, generated_signals)
        return np.mean(corr_coeff_scores)

    @staticmethod
    def compute_per_label(real_signals, generated_signals, labels):
        """
        Compute the Correlation Coefficient for each diagnostic label.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).
        labels : np.ndarray
            The diagnostic labels with shape (n_samples, n_labels).

        Returns
        -------
        dict
            A dictionary with labels as keys and Correlation Coefficient values as values.
        """
        n_labels = labels.shape[1]
        n_samples, n_leads, _ = real_signals.shape

        label_corr_coeff = {label: [] for label in range(n_labels)}

        for i in range(n_samples):
            for label in range(n_labels):
                if labels[i, label] == 1:
                    temp_scores = []
                    for j in range(n_leads):
                        temp_scores.append(CorrelationCoefficient.compute(real_signals[i, j], generated_signals[i, j]))
                    label_corr_coeff[label].append(np.mean(np.abs(temp_scores)))
        return label_corr_coeff
    
    @staticmethod
    def compute_per_label_agg(real_signals, generated_signals, labels):
        label_corr_coeff = CorrelationCoefficient.compute_per_label(real_signals, generated_signals, labels)
        label_corr_coeff = {label: np.mean(np.abs(corr_coeff_values)) for label, corr_coeff_values in label_corr_coeff.items() if corr_coeff_values}
        return label_corr_coeff


class MSE:
    """
    Class to compute the Mean Squared Error (MSE) between real and generated ECG signals.

    Methods
    -------
    compute(real_signal, generated_signal)
        Computes the MSE between the real and generated signals.
    compute_per_lead(real_signals, generated_signals)
        Computes the MSE for each lead.
    compute_aggregated(real_signals, generated_signals)
        Computes the aggregated MSE for all leads.
    compute_per_label(real_signals, generated_signals, labels)
        Computes the MSE for each diagnostic label.
    """

    @staticmethod
    def compute(real_signal, generated_signal):
        """
        Compute the Mean Squared Error (MSE) between real and generated ECG signals.

        Parameters
        ----------
        real_signal : np.ndarray
            The real ECG signal.
        generated_signal : np.ndarray
            The generated ECG signal.

        Returns
        -------
        float
            The MSE value.
        """
        return np.mean((real_signal - generated_signal) ** 2)

    @staticmethod
    def compute_per_lead(real_signals, generated_signals):
        """
        Compute the MSE for each lead.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        np.ndarray
            The MSE values for each lead.
        """
        n_samples, n_leads, _ = real_signals.shape
        mse_scores = np.zeros((n_samples, n_leads))

        for i in range(n_samples):
            for j in range(n_leads):
                mse_scores[i, j] = MSE.compute(real_signals[i, j], generated_signals[i, j])
        return mse_scores

    @staticmethod
    def compute_per_lead_agg(real_signals, generated_signals):
        mse_scores = MSE.compute_per_lead(real_signals, generated_signals)
        mse_scores = np.mean(mse_scores, axis = 0)
        return mse_scores

    @staticmethod
    def compute_aggregated(real_signals, generated_signals):
        """
        Compute the aggregated MSE for all leads.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        float
            The aggregated MSE value.
        """
        mse_scores = MSE.compute_per_lead_agg(real_signals, generated_signals)
        return np.mean(mse_scores)

    @staticmethod
    def compute_per_label(real_signals, generated_signals, labels):
        """
        Compute the MSE for each diagnostic label.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).
        labels : np.ndarray
            The diagnostic labels with shape (n_samples, n_labels).

        Returns
        -------
        dict
            A dictionary with labels as keys and MSE values as values.
        """
        n_labels = labels.shape[1]
        label_mse = {label: [] for label in range(n_labels)}

        for i in range(real_signals.shape[0]):
            for label in range(n_labels):
                if labels[i, label] == 1:
                    mse_value = MSE.compute(real_signals[i], generated_signals[i])
                    label_mse[label].append(mse_value)


        return label_mse
    
    @staticmethod
    def compute_per_label_agg(real_signals, generated_signals, labels):
        label_mse = MSE.compute_per_label(real_signals, generated_signals, labels)
        label_mse = {label: np.mean(mse_values) for label, mse_values in label_mse.items() if mse_values}
        return label_mse


class SNR:
    """
    Class to compute the Signal-to-Noise Ratio (SNR) between real and generated ECG signals.

    Methods
    -------
    compute(real_signal, generated_signal)
        Computes the SNR between the real and generated signals.
    compute_per_lead(real_signals, generated_signals)
        Computes the SNR for each lead.
    compute_aggregated(real_signals, generated_signals)
        Computes the aggregated SNR for all leads.
    compute_per_label(real_signals, generated_signals, labels)
        Computes the SNR for each diagnostic label.
    """

    @staticmethod
    def compute(real_signal, generated_signal):
        """
        Compute the Signal-to-Noise Ratio (SNR) between real and generated ECG signals.

        Parameters
        ----------
        real_signal : np.ndarray
            The real ECG signal.
        generated_signal : np.ndarray
            The generated ECG signal.

        Returns
        -------
        float
            The SNR value in dB.
        """
        signal_power = np.sum(real_signal ** 2)
        noise_power = np.sum((real_signal - generated_signal) ** 2)
        
        # Avoid division by zero by adding a small epsilon to noise_power if it is zero.
        if noise_power == 0:
            noise_power += 1e-10
        
        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def compute_per_lead(real_signals, generated_signals):
        """
        Compute the SNR for each lead.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        np.ndarray
            The SNR values for each lead.
        """
        n_samples, n_leads, _ = real_signals.shape
        snr_scores = np.zeros((n_samples, n_leads))

        for i in range(n_samples):
            for j in range(n_leads):
                snr_scores[i, j] = SNR.compute(real_signals[i, j], generated_signals[i, j])

        return snr_scores
    
    @staticmethod
    def compute_per_lead_agg(real_signals, generated_signals):
        snr_scores = SNR.compute_per_lead(real_signals, generated_signals)
        snr_scores = np.mean(snr_scores, axis = 0)
        return snr_scores

    @staticmethod
    def compute_aggregated(real_signals, generated_signals):
        """
        Compute the aggregated SNR for all leads.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).

        Returns
        -------
        float
            The aggregated SNR value.
        """
        snr_scores = SNR.compute_per_lead_agg(real_signals, generated_signals)
        return np.mean(snr_scores)

    @staticmethod
    def compute_per_label(real_signals, generated_signals, labels):
        """
        Compute the SNR for each diagnostic label.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).
        labels : np.ndarray
            The diagnostic labels with shape (n_samples, n_labels).

        Returns
        -------
        dict
            A dictionary with labels as keys and SNR values as values.
        """
        n_labels = labels.shape[1]
        label_snr = {label: [] for label in range(n_labels)}

        for i in range(real_signals.shape[0]):
            for label in range(n_labels):
                if labels[i, label] == 1:
                    snr_value = SNR.compute(real_signals[i], generated_signals[i])
                    label_snr[label].append(snr_value)


        return label_snr
    
    @staticmethod
    def compute_per_label_agg(real_signals, generated_signals, labels):
        label_snr = SNR.compute_per_label(real_signals, generated_signals, labels)
        label_snr = {label: np.mean(snr_values) for label, snr_values in label_snr.items() if snr_values}
        return label_snr


class SSIMSpectrogram:
    """
    Class to compute the Structural Similarity Index (SSIM) between the spectrograms of real and generated ECG signals.

    Methods
    -------
    compute_spectrogram(signal, fs=100)
        Computes the spectrogram of the given signal.
    compute_ssim(real_signal, generated_signal, fs=100)
        Computes the SSIM between the spectrograms of real and generated ECG signals.
    compute_per_lead(real_signals, generated_signals, fs=100)
        Computes the SSIM for each lead.
    compute_aggregated(real_signals, generated_signals, fs=100)
        Computes the aggregated SSIM for all leads.
    compute_per_label(real_signals, generated_signals, labels, fs=100)
        Computes the SSIM for each diagnostic label.
    """

    @staticmethod
    def compute_spectrogram(signal, fs=100):
        """
        Compute the spectrogram of the given signal.

        Parameters
        ----------
        signal : np.ndarray
            The ECG signal.
        fs : int, optional
            The sampling frequency of the signal. Default is 100.

        Returns
        -------
        tuple
            The frequencies, times, and spectrogram of the signal.
        """
        f, t, Sxx = spectrogram(signal, fs)
        return f, t, Sxx

    @staticmethod
    def compute_ssim(real_signal, generated_signal, fs=100):
        """
        Compute the SSIM between the spectrograms of real and generated ECG signals.

        Parameters
        ----------
        real_signal : np.ndarray
            The real ECG signal.
        generated_signal : np.ndarray
            The generated ECG signal.
        fs : int, optional
            The sampling frequency of the signals. Default is 100.

        Returns
        -------
        float
            The SSIM value between the spectrograms of the real and generated signals.
        """
        _, _, Sxx_real = SSIMSpectrogram.compute_spectrogram(real_signal, fs)
        _, _, Sxx_generated = SSIMSpectrogram.compute_spectrogram(generated_signal, fs)
        
        # Normalize spectrograms to [0, 1] range for SSIM calculation
        Sxx_real = (Sxx_real - np.min(Sxx_real)) / (np.max(Sxx_real) - np.min(Sxx_real))
        Sxx_generated = (Sxx_generated - np.min(Sxx_generated)) / (np.max(Sxx_generated) - np.min(Sxx_generated))
        
        return ssim(Sxx_real, Sxx_generated, data_range = 2.0, channel_axis = 1, win_size = 3)

    @staticmethod
    def compute_per_lead(real_signals, generated_signals, fs=100):
        """
        Compute the SSIM for each lead.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).
        fs : int, optional
            The sampling frequency of the signals. Default is 100.

        Returns
        -------
        np.ndarray
            The SSIM values for each lead.
        """
        n_samples, n_leads, _ = real_signals.shape
        ssim_scores = np.zeros((n_samples, n_leads))

        for i in range(n_samples):
            for j in range(n_leads):
                ssim_scores[i, j] = SSIMSpectrogram.compute_ssim(real_signals[i, j], generated_signals[i, j], fs)
        return ssim_scores
    
    @staticmethod
    def compute_per_lead_agg(real_signals, generated_signals, fs=100):
        ssim_scores = SSIMSpectrogram.compute_per_lead(real_signals, generated_signals, fs)
        ssim_scores = np.mean(ssim_scores, axis = 0)
        return ssim_scores


    @staticmethod
    def compute_aggregated(real_signals, generated_signals, fs=100):
        """
        Compute the aggregated SSIM for all leads.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).
        fs : int, optional
            The sampling frequency of the signals. Default is 100.

        Returns
        -------
        float
            The aggregated SSIM value.
        """
        ssim_scores = SSIMSpectrogram.compute_per_lead_agg(real_signals, generated_signals, fs)
        return np.mean(ssim_scores)

    @staticmethod
    def compute_per_label(real_signals, generated_signals, labels, fs=100):
        """
        Compute the SSIM for each diagnostic label.

        Parameters
        ----------
        real_signals : np.ndarray
            The real ECG signals with shape (n_samples, n_leads, n_timestamps).
        generated_signals : np.ndarray
            The generated ECG signals with shape (n_samples, n_leads, n_timestamps).
        labels : np.ndarray
            The diagnostic labels with shape (n_samples, n_labels).
        fs : int, optional
            The sampling frequency of the signals. Default is 100.

        Returns
        -------
        dict
            A dictionary with labels as keys and SSIM values as values.
        """
        n_labels = labels.shape[1]
        label_ssim = {label: [] for label in range(n_labels)}

        for i in range(real_signals.shape[0]):
            for label in range(n_labels):
                if labels[i, label] == 1:
                    ssim_value = SSIMSpectrogram.compute_ssim(real_signals[i], generated_signals[i], fs)
                    label_ssim[label].append(ssim_value)

        return label_ssim
    
    @staticmethod
    def compute_per_label_agg(real_signals, generated_signals, labels, fs=100):
        label_ssim = SSIMSpectrogram.compute_per_label(real_signals, generated_signals, labels, fs)
        label_ssim = {label: np.mean(ssim_values) for label, ssim_values in label_ssim.items() if ssim_values}
        return label_ssim

class MelSpectrogram:
    """
    Class to compute the L1 Spectrogram distances between the real and generated ECG signals.
    """
    
    @staticmethod
    def compute_aggregated(real_signals, generated_signals, sample_rate = 100):
        """
        Compute the L1 Spectrogram distances for all 12 lead signals
        """
        ecg_sig_real = ECGSignal(real_signals, sample_rate=sample_rate)
        ecg_sig_generated = ECGSignal(generated_signals, sample_rate=sample_rate)
        mel_loss = MelSpectrogramLoss(window_lengths=[64], n_mels=[16], loss_fn=torch.nn.L1Loss())
        loss = mel_loss(ecg_sig_real, ecg_sig_generated)

        return loss
    
    @staticmethod
    def compute_per_lead(real_signals, generated_signals, sample_rate = 100):
        """
        Compute the L1 Spectrogram distances for each lead
        """
        n_samples, n_leads, _ = real_signals.shape
        mel_scores = np.zeros((n_samples, n_leads))

        mel_loss = MelSpectrogramLoss(window_lengths=[64], n_mels=[16], loss_fn=torch.nn.L1Loss())

        for i in range(n_samples):
            for j in range(n_leads):
                ecg_sig_real = ECGSignal(real_signals[i,j,:], sample_rate=sample_rate)
                ecg_sig_generated = ECGSignal(generated_signals[i,j,:], sample_rate=sample_rate)
                mel_scores[i, j] = mel_loss(ecg_sig_real, ecg_sig_generated)

        return mel_scores
    
    @staticmethod
    def compute_per_lead_agg(real_signals, generated_signals, sample_rate = 100):
        n_samples, n_leads, _ = real_signals.shape
        mel_loss = MelSpectrogramLoss(window_lengths=[64], n_mels=[16], loss_fn=torch.nn.L1Loss())
        mel_scores = np.zeros(n_leads)

        for j in range(n_leads):
            ecg_sig_real = ECGSignal(real_signals[:,j,:], sample_rate=sample_rate)
            ecg_sig_generated = ECGSignal(generated_signals[:,j,:], sample_rate=sample_rate)
            mel_scores[j] = mel_loss(ecg_sig_real, ecg_sig_generated)
        return mel_scores

    
    @staticmethod
    def compute_per_label(real_signals, generated_signals, labels, sample_rate = 100):
        """
        Compute the L1 Spectrogram distances for each label
        """
        n_labels = labels.shape[1]
        mel_scores = {label: [] for label in range(n_labels)}

        mel_loss = MelSpectrogramLoss(window_lengths=[64], n_mels=[16], loss_fn=torch.nn.L1Loss())
        
        for i in range(real_signals.shape[0]):
            for label in range(n_labels):
                # Extract samples corresponding to the current label as matrices
                ecg_sig_real = ECGSignal(real_signals[i], sample_rate=sample_rate)
                ecg_sig_generated = ECGSignal(generated_signals[i], sample_rate=sample_rate)
                mel_scores[label].append(mel_loss(ecg_sig_real, ecg_sig_generated))
        
        return mel_scores
    
    @staticmethod
    def compute_per_label_agg(real_signals, generated_signals, labels, sample_rate = 100):
        n_labels = labels.shape[1]

        mel_scores = {}
        mel_loss = MelSpectrogramLoss(window_lengths=[64], n_mels=[16], loss_fn=torch.nn.L1Loss())

        for label in range(n_labels):
            # Find indices of samples corresponding to the current label
            indices = np.where(labels[:, label] == 1)[0]

            if len(indices) > 0:
                # Extract samples corresponding to the current label as matrices
                ecg_sig_real = ECGSignal(real_signals[indices], sample_rate=sample_rate)
                ecg_sig_generated = ECGSignal(generated_signals[indices], sample_rate=sample_rate)
                mel_scores[label] = mel_loss(ecg_sig_real, ecg_sig_generated)
        return mel_scores
