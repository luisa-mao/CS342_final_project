---
{{ card_data }}
---

# Model Card for Conditional Latent Diffusion Model

This is a latent diffusion model that accepts text embeddings as input and outputs RGB images. We use CLIP embeddings for training.

## Model Details

This model does latent diffusion, starting with a noise vector in the latent dimension, and using a UNet to iteratively denoise the vector, before putting it through a decoder to go back to pixel space. Cross attention layers are used to condition the denoising on the text prompt, which also goes through an encoder.

### Model Description

- **Developed by:** Andrew McAlinden and Luisa Mao
- **Model type:** UNet Transformer
- **Language(s) (NLP):** English
- **License:** None. Open to all.

### Model Sources [optional]

- **Repository:** https://github.com/CompVis/latent-diffusion
- **Paper:** https://arxiv.org/pdf/2112.10752

## Uses

### Direct Use

This model would be well used by advertizing agencies to generate quick and effective images for ads.

### Downstream Use [optional]

This model could be used by artists to get ideas of scenes to draw that match a prompt, or to speed up the artistic process.

### Out-of-Scope Use

This model should not be used for any task where the cost of failure is high, such as generating an image of an airplane safety waiver.

## Bias, Risks, and Limitations

The model is only trained on a animal dataset, so it isn't good at generating images of anything that isn't an animal. 

### Recommendations

Use this model to generate low quality images of animals, and don't expect too much.

## How to Get Started with the Model

Use the code in this repo to get started with the model.

## Training Details

### Training Data

See dataset_card.md. We use this dataset: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset

### Training Procedure

We take the top 11 classes most common classes of animals and create a dataset with them. This is about 7500 images total. Then, train for 60 epochs. Use a fixed VAE for both the images and prompts.

#### Training Hyperparameters

image_size = 128
num_classes = 11
batch_size = 128
num_ddpm_steps = 1000
num_epochs = 60
lr = 1e-4
num_warmup_steps = 500
scale_factor= 0.18215
seed = 423432

## Evaluation

We evaluate the model using the FID metric as well as precision and recall. 

### Testing Data, Factors & Metrics

#### Testing Data

Since we want to know how well our model learned the distribution of the data it was trained on, we re-use the train data.

#### Metrics

The FID and precision/recall metrics each measure how well the model learned the distribution of its training data, which is what we want to see for diffusion. 

### Results

With k=10, we get precision of .55 and recall of .225. Our FID-score is 172.5.

#### Summary

We are content with the performance of our model given the small training dataset.

## Environmental Impact

- **Hardware Type:** Single Nvidia A100
- **Hours used:** 1
- **Cloud Provider:** Private
- **Compute Region:** North America
- **Carbon Emitted:** 0.11

#### Software

PyTorch, NumPy, HuggingFace

## Model Card Contact

@andrewmcalinden on GitHub