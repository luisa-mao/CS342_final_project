from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import os
import sys
parent_dir = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(parent_dir)
from tqdm import tqdm
import cv2
import numpy as np



class TinyImageNetDataset(Dataset):
    def __init__(self, dataset_path = 'tiny-imagenet-200', mode = 'train'):
        super().__init__()
        # clip feature length is 768
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
        self.mode = mode
        self.wnids_path = os.path.join(parent_dir, dataset_path, "wnids.txt")
        self.words_path = os.path.join(parent_dir, dataset_path, "words.txt")
        self.images_path = os.path.join(parent_dir, dataset_path, mode)
        self.wnid_to_images = {}
        self.image_to_wnid = []
        self.image_names = []
        self.len = 0

        # Load wnids.txt (200 valid classes)
        with open(self.wnids_path, 'r') as f:
            valid_wnids = set(f.read().splitlines())

        self.wnid_to_word = {k:[] for k in valid_wnids}
        self.wnid_to_text_embedding = {k:None for k in valid_wnids}

        # Load words.txt and filter only valid wnids
        # pass words through clip and pad
        with open(self.words_path, 'r') as f:
            for line in tqdm(f, desc = "Loading text"):
                wnid, description = line.split('\t')
                if wnid in valid_wnids:
                    words = description.split(',')
                    self.wnid_to_word[wnid] += words
                    embeds = self.text_enc(words).detach()
                    self.wnid_to_text_embedding[wnid] = embeds


        for dir in tqdm(os.listdir(self.images_path), desc = 'Loading Images'):
            if dir == 'n01443537': # change later
                image_names = os.listdir(os.path.join(self.images_path, dir, 'images'))
                self.wnid_to_images[dir] = image_names
                self.image_to_wnid += [dir] * len(image_names)
                self.image_names += os.listdir(os.path.join(self.images_path, dir, 'images'))
                self.len+=len(image_names)

    def text_enc(self, prompts, maxlen=None):
        '''
        A function to take a texual prompt and convert it into embeddings
        '''
        if maxlen is None: maxlen = self.tokenizer.model_max_length
        inp = self.tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt") 
        return self.text_encoder(inp.input_ids)[0] #.half()


    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        wnid = self.image_to_wnid[idx]
        word_idx = np.random.randint(0, len(self.wnid_to_text_embedding[wnid]))
        word = self.wnid_to_word[wnid][word_idx]
        text_embedding = self.wnid_to_text_embedding[wnid][word_idx]
        image_path = os.path.join(self.images_path, wnid, 'images', self.image_names[idx])
        image = cv2.imread(image_path)
        image = image.astype(np.float32) / 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1)
        return image, text_embedding, wnid, word # wnid and word are for debugging

if __name__ == '__main__':
    dataset = TinyImageNetDataset()
    print(len(dataset))
    image, text_embedding, wnid, word = dataset[432]
    print(image.shape, text_embedding.shape, wnid, word)
