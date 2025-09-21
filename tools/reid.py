# tools/reid.py (New version using PyTorch Hub)
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image

class ReidFeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        # Load a pre-trained ResNet-50 model
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        
        # We use a hook to grab the output from the penultimate layer (avgpool)
        self.features = None
        def hook(module, input, output):
            self.features = output.squeeze()

        model.avgpool.register_forward_hook(hook)
        
        # Set model to evaluation mode and move to the correct device
        self.model = model.to(device).eval()
        
        # Define the standard ImageNet transformations
        self.preprocess = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Re-ID model (ResNet-50 from PyTorch Hub) loaded.")

    @torch.no_grad()
    def __call__(self, image_crops):
        """
        Extracts features from a list of image crops.
        Args:
            image_crops (list of np.ndarray): A list of images (bounding box crops from OpenCV).
        Returns:
            torch.Tensor: A tensor containing the feature embeddings.
        """
        if not image_crops:
            return torch.empty((0, 2048), device=self.device) # ResNet-50 features are 2048-dim
        
        # Convert OpenCV BGR crops to PIL RGB and apply transformations
        batch = []
        for crop in image_crops:
            if crop.size == 0: continue
            pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            processed_tensor = self.preprocess(pil_image)
            batch.append(processed_tensor)
        
        if not batch:
            return torch.empty((0, 2048), device=self.device)

        # Stack the tensors into a single batch and move to the device
        batch_tensor = torch.stack(batch).to(self.device)
        
        # Run the model
        self.model(batch_tensor)
        
        # The hook has now captured the features
        return self.features.clone()

# Need to import cv2 here for the conversion in __call__
import cv2
