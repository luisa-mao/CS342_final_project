import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pretrained InceptionV3 model
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()
        inception.eval()
        self.features = inception
        self.means = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape: [1, 3, 1, 1]
        self.stds = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # Shape: [1, 3, 1, 1]        
        # Freeze the model parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.resize(x, (299, 299))  
        x = x-self.means
        x = x/self.stds      
        x = self.features(x)  # Output: [B, 2048, 1, 1]
        x = torch.flatten(x, 1)  # Flatten to [B, 2048]
        
        return x
