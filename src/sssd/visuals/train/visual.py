import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_generated_audio(audio_path, save_dir=None, num_samples=5):
    """
    Visualize generated ECG audio data from a .npy file.
    
    Args:
        audio_path (str): Path to the generated_audio12.npy file
        save_dir (str, optional): Directory to save the visualization plots. If None, plots will be displayed.
        num_samples (int): Number of samples to visualize
    """
    # Load the generated audio data
    generated_audio = np.load(audio_path)
    
    # Create figure with subplots for each sample
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i in range(num_samples):
        if i < len(generated_audio):
            # Plot all 8 leads
            for lead in range(8):
                axes[i].plot(generated_audio[i, lead], label=f'Lead {lead+1}')
            
            axes[i].set_title(f'Sample {i+1}')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Amplitude')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'generated_audio_visualization.png'))
        plt.close()
    else:
        plt.show()

def visualize_audio_comparison(generated_audio_path, real_audio_path, save_dir=None, num_samples=5):
    """
    Visualize and compare generated ECG audio data with real ECG audio data.
    
    Args:
        generated_audio_path (str): Path to the generated_audio12.npy file
        real_audio_path (str): Path to the real_audio.npy file
        save_dir (str, optional): Directory to save the visualization plots. If None, plots will be displayed.
        num_samples (int): Number of samples to visualize
    """
    # Load both audio datasets
    generated_audio = np.load(generated_audio_path)
    real_audio = np.load(real_audio_path)
    
    # Create figure with subplots for each sample
    fig, axes = plt.subplots(num_samples, 2, figsize=(20, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each sample
    for i in range(num_samples):
        if i < len(generated_audio) and i < len(real_audio):
            # Plot generated audio
            for lead in range(8):
                axes[i, 0].plot(generated_audio[i, lead], label=f'Lead {lead+1}')
            axes[i, 0].set_title(f'Generated Sample {i+1}')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].legend()
            axes[i, 0].grid(True)
            
            # Plot real audio
            for lead in range(8):
                axes[i, 1].plot(real_audio[i, lead], label=f'Lead {lead+1}')
            axes[i, 1].set_title(f'Real Sample {i+1}')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].legend()
            axes[i, 1].grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'audio_comparison_visualization.png'))
        plt.close()
    else:
        plt.show()

def visualize_audio_comparison_overlay(generated_audio_path, real_audio_path, save_dir=None, num_samples=5):
    """
    Visualize and compare generated ECG audio data with real ECG audio data by overlaying them.
    
    Args:
        generated_audio_path (str): Path to the generated_audio12.npy file
        real_audio_path (str): Path to the real_audio.npy file
        save_dir (str, optional): Directory to save the visualization plots. If None, plots will be displayed.
        num_samples (int): Number of samples to visualize
    """
    # Load both audio datasets
    generated_audio = np.load(generated_audio_path)
    real_audio = np.load(real_audio_path)
    
    # Create figure with subplots for each sample
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    # Plot each sample
    for i in range(num_samples):
        if i < len(generated_audio) and i < len(real_audio):
            # Plot both generated and real audio
            for lead in range(8):
                axes[i].plot(generated_audio[i, lead], '--', alpha=0.7, label=f'Generated Lead {lead+1}')
                axes[i].plot(real_audio[i, lead], '-', alpha=0.7, label=f'Real Lead {lead+1}')
            
            axes[i].set_title(f'Sample {i+1} Comparison')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel('Amplitude')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'audio_comparison_overlay_visualization.png'))
        plt.close()
    else:
        plt.show()

def visualize_sample_lead_comparison(generated_audio_path, real_audio_path, save_dir=None, num_samples=5):
    """
    Visualize each sample and lead separately for detailed comparison.
    
    Args:
        generated_audio_path (str): Path to the generated_audio12.npy file
        real_audio_path (str): Path to the real_audio.npy file
        save_dir (str, optional): Directory to save the visualization plots. If None, plots will be displayed.
        num_samples (int): Number of samples to visualize
    """
    # Load both audio datasets
    generated_audio = np.load(generated_audio_path)
    real_audio = np.load(real_audio_path)
    
    # Create a directory for individual plots if save_dir is provided
    if save_dir:
        individual_plots_dir = os.path.join(save_dir, 'individual_plots')
        os.makedirs(individual_plots_dir, exist_ok=True)
    
    # Plot each sample and lead separately
    for i in range(num_samples):
        if i < len(generated_audio) and i < len(real_audio):
            for lead in range(8):
                plt.figure(figsize=(15, 5))
                
                # Plot both generated and real audio for this lead
                plt.plot(generated_audio[i, lead], '--', label='Generated', alpha=0.7)
                plt.plot(real_audio[i, lead], '-', label='Real', alpha=0.7)
                
                plt.title(f'Sample {i+1}, Lead {lead+1} Comparison')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)
                
                if save_dir:
                    plt.savefig(os.path.join(individual_plots_dir, f'sample_{i+1}_lead_{lead+1}.png'))
                    plt.close()
                else:
                    plt.show()

if __name__ == "__main__":
    # Example usage
    generated_audio_path = "synth_audio.npy"
    real_audio_path = "real_audio.npy"
    save_dir = "./"
    
    # Visualize side by side comparison
    visualize_audio_comparison(generated_audio_path, real_audio_path, save_dir)
    
    # Visualize overlay comparison
    visualize_audio_comparison_overlay(generated_audio_path, real_audio_path, save_dir)
    
    # Visualize individual sample and lead comparisons
    visualize_sample_lead_comparison(generated_audio_path, real_audio_path, save_dir)
