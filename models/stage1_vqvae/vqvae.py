import torch
import pytorch_lightning as pl
from functools import partial

from models.stage1_vqvae.EncoderTriple import TripleGrainEncoder
from models.stage1_vqvae.DecoderPositional import Decoder
from models.stage1_vqvae.quantize2_mask import VectorQuantize2

class TripleGrainVQModel(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 feature_ch=256,
                 vae_codebook_size = 2048,
                 vae_dim=512,
                 resolution = 128,
                 quant_sample_temperature = 0., 
                 ckpt_path = None,
                 ignore_keys = [],
                 ):
        super().__init__()
        self.encoder =TripleGrainEncoder(ch = feature_ch, ch_mult=[1,1,2,2,4], num_res_blocks=2, attn_resolutions=[8,16,32], dropout=0.0, resamp_with_conv=True, in_channels = in_channels, resolution=resolution, z_channels=vae_dim)
        self.decoder = Decoder(ch=feature_ch, in_ch = vae_dim, out_ch=in_channels, ch_mult=[1,1,2], num_res_blocks=2, resolution=resolution, attn_resolutions=[32],latent_size=32, window_size=2)
        self.quantize = VectorQuantize2(codebook_size=vae_codebook_size, codebook_dim=vae_dim, channel_last=False, accept_image_fmap=False, commitment_beta=0.25, decay=0.99, restart_unused_codes=True)
        self.quant_conv = torch.nn.Conv1d(vae_dim, vae_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(vae_dim, vae_dim, 1)
        self.quant_sample_temperature = quant_sample_temperature

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h_dict = self.encoder(x, None)  
        h = h_dict["h_triple"]        
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(x=h, temp=self.quant_sample_temperature, codebook_mask=None) 
        return quant, emb_loss, info

    def decode(self, quant, grain_indices=None):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, grain_indices=None)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.get_codebook_entry(code_b)
        dec = self.decode(quant_b.transpose(1,2))
        return quant_b,dec

    def forward(self, input):
        quant, diff, _= self.encode(input)
        dec = self.decode(quant, grain_indices=None)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.size(1) != 3:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x


if __name__ == "__main__":
    pass