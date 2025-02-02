import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from piq import information_fidelity_criterion as ifc, normalized_quality_measure as nqm

class ImageEvaluator:
    def __init__(self):
        self.metrics = {'PSNR': [], 'SSIM': [], 'IFC': [], 'NQM': [], 'WPSNR': [], 'MSSIM': []}
    
    def evaluate(self, hr, ouput):
        
        """Evaluates the HR vs. Output images (expects PyTorch tensors)."""
        hr = hr.squeeze(0).cpu()
        output = output.squeeze(0).cpu().clamp(0, 1)
        
        self.metrics['PSNR'].append(psnr(hr.numpy(), output.numpy()))
        self.metrics['SSIM'].append(ssim(hr.numpy(), output.numpy(), multichannel=True))
        self.metrics['IFC'].append(ifc(hr.unsqueeze(0), output.unsqueeze(0)).item())
        self.metrics['NQM'].append(nqm(hr.unsqueeze(0), output.unsqueeze(0)).item())
        self.metrics['WPSNR'].append(ImageEvaluator.wpsnr(hr, output))
        self.metrics['MSSIM'].append(ImageEvaluator.mssim(hr, output))
        self.metrics['MSE'].append(F.mse_loss(output, hr).item())
    
    def compute_averages(self):
        return {metric: np.mean(values) for metric, values in self.metrics.items()}
    
    @staticmethod
    def wpsnr(original, reconstructed):
        """Weighted Peak Signal-to-Noise Ratio (WPSNR)."""
        weight = torch.ones_like(original)  # For simplicity, uniform weight
        mse = torch.mean(((original - reconstructed) ** 2) * weight)
        return 10 * torch.log10(1 / mse) if mse > 0 else float('inf')

    @staticmethod
    def mssim(original, reconstructed):
        """Mean Structural Similarity Index (MSSIM)."""
        original_np = original.permute(1, 2, 0).cpu().numpy()
        reconstructed_np = reconstructed.permute(1, 2, 0).cpu().numpy()
        return ssim(original_np, reconstructed_np, multichannel=True, gaussian_weights=True)