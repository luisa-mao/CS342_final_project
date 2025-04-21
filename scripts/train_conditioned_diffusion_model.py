import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL 


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # frozen pretrained autoencoder
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float32)
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.unet = UNet2DConditionModel(cross_attention_dim = (768),
                                    down_block_types = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
                                    block_out_channels= (320, 640, 1280),
                                    up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                                    ).to(torch.float32)
        self.unet.requires_grad_(True)

    def forward(self, images, condition_embeds, ts):
        init_image = images * 2.0 - 1.0

        # convert to float32
        init_image = init_image.to(dtype=torch.float32)
        
        # Encode the image to latents using the VAE
        img_latents = self.vae.encode(init_image).latent_dist.sample() * 0.18215
        conditioned_latents = self.unet(img_latents, timestep = ts, encoder_hidden_states = condition_embeds).sample
        conditioned_latents = (1 / 0.18215) * conditioned_latents  
        outputs = self.vae.decode(conditioned_latents).sample     
        outputs = (outputs / 2 + 0.5)
        return outputs
    
def train(num_epochs, dataloader, text_embedding, optimizer, mse_loss):
    # this is pseudocode from chatgpt. it wont run yet
    # maybe this is a better resource https://huggingface.co/docs/diffusers/en/tutorials/basic_training
    for epoch in range(num_epochs):
        for x0, text in dataloader:  # x0: image, text: string
            
            # 1. Encode the text
            text_embedding = text_encoder(text)  # shape: (batch_size, embed_dim)

            # 2. Sample random timestep t for each example
            t = sample_uniform_integers(1, T, size=(batch_size,))

            # 3. Sample Gaussian noise
            noise = sample_normal(shape=x0.shape)

            # 4. Forward diffusion to get x_t
            alpha_bar_t = alphas_cumprod[t]  # shape: (batch_size,)
            sqrt_alpha_bar_t = sqrt(alpha_bar_t).reshape(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = sqrt(1 - alpha_bar_t).reshape(-1, 1, 1, 1)
            x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

            # 5. Predict the noise using the model
            # The model should accept (x_t, t, text_embedding)
            epsilon_hat = model(x_t, t, text_embedding)

            # 6. Compute the loss (simple MSE)
            loss = mse_loss(epsilon_hat, noise)

            # 7. Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    model = DiffusionModel()
    input = torch.randn((5, 3, 256, 256))
    cond_embed = torch.randn((5, 77, 768))
    out = model(input, cond_embed, 1)
    print(out.shape)




