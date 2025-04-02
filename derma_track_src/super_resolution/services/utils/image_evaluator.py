import torch

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import ssim, ms_ssim
from super_resolution.services.utils.running_average import RunningAverage

class ImageEvaluator:
    
    def __init__(self):
        
        self.metrics = {'MSE': RunningAverage(), 'PSNR': RunningAverage(), 'SSIM': RunningAverage(), 'MSSIM': RunningAverage(),'LPIPS': RunningAverage()}
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type = 'alex', normalize = True)
        
    def evaluate(self, hr: torch.Tensor, output: torch.Tensor) -> None:
        """
        Evaluation between original and generate image on multiple metrics

        Args:
            hr (torch.Tensor): The original image
            output (torch.Tensor): The generate image
        """
        
        clamp_hr = torch.clamp(hr, 0.0, 1.0)
        
        clamp_output = torch.clamp(output, 0.0, 1.0)
        
        self.metrics['MSE'].update(value = self.__mse(hr = clamp_hr, output = clamp_output))
        
        self.metrics['PSNR'].update(value = self.__psnr())
        
        self.metrics['SSIM'].update(value = self.__ssim(hr = clamp_hr, output = clamp_output))
        
        self.metrics['MSSIM'].update(value = self.__mssim(hr = clamp_hr, output = clamp_output))
        
        self.metrics['LPIPS'].update(value = self.__lpips(hr = clamp_hr, output = clamp_output))
    
    def get_average_metrics(self) -> dict:
        
        """
        Returns the mean of all stored metric values.

        Returns:
            dict: Dictionary with average values of each metric.
        """
        return {metric: values.average for metric, values in self.metrics.items()}
    
    def __mse(self, hr: torch.Tensor, output: torch.Tensor) -> float:
        """
        Computes the Mean Squared Error between two images.

        Args:
            hr (torch.Tensor): The original image
            output (torch.Tensor): The generate image

        Returns:
            float: MSE value (from [0, inf) -> low better)
        """
        
        return torch.mean((hr - output) ** 2).item()
    
    def __psnr(self) -> float:
        """
        Computes the Peak Signal-to-Noise Ratio between two images.

        Returns:
            float: PSNR value (from [0, 100] -> higher better)
        
        More Informations:
            In the original formula, the numerator is suppose to be the max value of a pixel,
            but we normalize the value so the max value is 1 instead of 255.
        """
        
        mse = self.metrics['MSE'].all_values[-1]
        
        if (mse == 0):
            return 100
        
        return (10 * torch.log10(torch.tensor(1.0) / mse)).item()
    
    def __ssim(self, hr: torch.Tensor, output: torch.Tensor) -> float:
        """
        Computes the Structural Similarity Index between two images.

        Args:
            hr (torch.Tensor): The original image
            output (torch.Tensor): The generate image

        Returns:
            float: SSIM value (from [-1, 1] -> where 1 = identical)
        """

        return ssim(X = hr, Y = output, data_range = 1.0).item()
        
    
    def __mssim(self, hr: torch.Tensor, output: torch.Tensor) -> float:
        """
        Computes the Mean Structural Similarity Index between two images.

        Args:
            hr (torch.Tensor): The original image
            output (torch.Tensor): The generate image

        Returns:
            float: MSSIM value (from [-1, 1] -> higher better)
        """
        return ms_ssim(X = hr, Y = output, data_range = 1.0).item()
    
    def __lpips(self, hr: torch.Tensor, output: torch.Tensor) -> float:
        """
        Computes the Learned Perceptual Image Patch Similarity between two images.

        Args:
            hr (torch.Tensor): The original image
            output (torch.Tensor): The generate image

        Returns:
            float: LPIPS value (from [0,1] -> lower better)
        """

        return self.lpips_loss(img1 = hr, img2 = output).item()