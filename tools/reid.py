# tools/reid.py
import torch
import torchvision.transforms as T
from torchreid.utils import FeatureExtractor

class ReidFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        # This will download the model weights automatically the first time
        self.extractor = FeatureExtractor(
            model_name='osnet_x0_25',
            model_path='osnet_x0_25_market1501.pt', # A popular lightweight model
            device=device
        )
        print("Re-ID model (OSNet) loaded.")

    def __call__(self, image_crops):
        """
        Extracts features from a list of image crops.
        Args:
            image_crops (list of np.ndarray): A list of images (bounding box crops).
        Returns:
            torch.Tensor: A tensor of shape (N, 512) containing the feature embeddings.
        """
        if not image_crops:
            return torch.empty((0, 512), device=self.device)
        
        # The extractor expects a list of numpy arrays (H, W, C) with BGR channels
        features = self.extractor(image_crops)
        return features
