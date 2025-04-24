import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
import numpy as np

class AnimalDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, classes, img_size, device=None):
        super().__init__()
        self.root_dir = root_dir
        self.classes = classes
        self.samples = [] # List of (image_path, label)
        self.img_size = img_size
        self.device = device 

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
        self.text_encoder.eval()
        if device is not None:
            self.text_encoder.to(device)

        # can batch this if slow
        self.label_to_text_embedding = {}
        with torch.no_grad():
            for label in classes:
                # inputs = self.tokenizer(label, padding="max_length", max_length=self.tokenizer.model_max_length, 
                #                 truncation=True, return_tensors="pt")
                # inputs = {key: value.to(device) for key, value in inputs.items()}
                # text_embedding = self.text_encoder(**inputs).pooler_output
                text_embedding = self.get_embedding(label)
                self.label_to_text_embedding[label] = text_embedding

        self._build_samples_list()

    def get_embedding(self, label):
        inputs = self.tokenizer(label, padding="max_length", max_length=self.tokenizer.model_max_length, 
                                    truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        text_embedding = self.text_encoder(**inputs).pooler_output
        return text_embedding

    def _build_samples_list(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, class_name))

    def __len__(self):
        return len(self.samples)

    # returns (tensor image, tensor text embedding)
    # image will be on [-1, 1]
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        # convert image to tensor
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        image = transform(image)        
        embedding = self.label_to_text_embedding[label]
        return image, embedding

