import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from models.stage2_retnet.xpos_relative_position import XPOS
from models.stage2_retnet.retnet import RetNet

def mask_latents(latents,mask_ratio):
    mask = torch.bernoulli(mask_ratio * torch.ones(latents.shape, device=latents.device))
    mask = mask.round().to(dtype=torch.int64)
    indices = torch.zeros_like(latents)  
    indices = mask*latents+(1-mask)*indices
    return indices

class StackedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StackedNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU(True)
        self.layer2 = nn.Linear(input_size, hidden_size)
        self.relu2 = nn.LeakyReLU(True)
        self.layer3 = nn.Linear(input_size, hidden_size)
        self.relu3 = nn.LeakyReLU(True)
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, x1, x2, x3):
        out1 = self.relu1(self.layer1(x1))
        out2 = self.relu2(self.layer2(x2))
        out3 = self.relu3(self.layer3(x3))
        stacked_feature = torch.cat((out1, out2, out3), dim=2)
        output = self.fc(stacked_feature)
        return output

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)  
    
class Whole2Part(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(Whole2Part, self).__init__()
        self.ln_f = nn.LayerNorm(input_size)        
        self.head = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(True),
                nn.Linear(hidden_size, output_size)
            )
        
    def forward(self, x):
        part = self.head(self.ln_f(x))
        return part
        
class Audio2Motion_RetNet(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                n_classes = 25):
        super().__init__()
        self.num_vq=num_vq 
        self.layers = num_layers
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.chunk_size = 2
        self.body_emb = nn.Embedding(self.num_vq, self.embed_dim)
        self.lhand_emb = nn.Embedding(self.num_vq, self.embed_dim)
        self.rhand_emb = nn.Embedding(self.num_vq, self.embed_dim)
        
        self.class_cond_embedding = nn.Embedding(
            n_classes, self.embed_dim*2
        )
        self.gate = GatedActivation()
        self.stack = StackedNet(self.embed_dim*2,self.embed_dim*2,self.embed_dim*2)
    
        self.block = RetNet(self.layers, 256, 512, self.n_head) 

        self.body_logits = Whole2Part(self.embed_dim,self.num_vq//2,self.num_vq)  
        self.lhand_logits = Whole2Part(self.embed_dim,self.num_vq//2,self.num_vq)  
        self.rhand_logits = Whole2Part(self.embed_dim,self.num_vq//2,self.num_vq)   
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, body_latents, lhand_latents, rhand_latents, audio, id):
        
        h = self.class_cond_embedding(id)    
        body_embeddings = torch.cat([self.body_emb(body_latents),audio.transpose(1,2)],dim=2) 
        lhand_embeddings = torch.cat([self.lhand_emb(lhand_latents),audio.transpose(1,2)],dim=2)
        rhand_embeddings = torch.cat([self.rhand_emb(rhand_latents),audio.transpose(1,2)],dim=2)
        whole_embeddings = self.stack(body_embeddings,lhand_embeddings,rhand_embeddings) 
        
        x = self.gate(whole_embeddings.transpose(1,2) + h[:, :, None]).transpose(1,2)             
        x = self.block(x) 
        body_logits = self.body_logits(x)
        lhand_logits = self.lhand_logits(x)
        rhand_logits = self.rhand_logits(x)
        
        return body_logits,lhand_logits,rhand_logits
    
    def sample_chunkwise(self,audio,id,chunk_size=30):
        
        body_latents=torch.zeros((1, audio.shape[2]),dtype=torch.int64, device=audio.device)
        body_latents[:,:1] = torch.randint(low=0,high=2023,size=(1,1))
        lhand_latents=torch.zeros((1, audio.shape[2]),dtype=torch.int64, device=audio.device)
        lhand_latents[:,:1] = torch.randint(low=0,high=2023,size=(1,1))
        rhand_latents=torch.zeros((1, audio.shape[2]),dtype=torch.int64, device=audio.device)
        rhand_latents[:,:1] = torch.randint(low=0,high=2023,size=(1,1))
        
        h = self.class_cond_embedding(id)  
          
        body_embeddings = torch.cat([self.body_emb(body_latents),audio.transpose(1,2)],dim=2) ###[bs, 32, 256]
        lhand_embeddings = torch.cat([self.lhand_emb(lhand_latents),audio.transpose(1,2)],dim=2)
        rhand_embeddings = torch.cat([self.rhand_emb(rhand_latents),audio.transpose(1,2)],dim=2)
        whole_embeddings = self.stack(body_embeddings,lhand_embeddings,rhand_embeddings) ###[512, 32, 512]
        
        X = self.gate(whole_embeddings.transpose(1,2) + h[:, :, None]).transpose(1,2)  ###[512, 32, 256]              
        r_n_1s = [
            [
                torch.zeros(self.block.hidden_dim// self.n_head, self.block.v_dim // self.n_head).unsqueeze(0).repeat(1, 1, 1).to(audio.device)
                for _ in range(self.n_head)
            ]
            for _ in range(self.layers)
        ]
        Y_chunkwise = []
        
        for i in range(audio.shape[2]// chunk_size+1):
            y_n, r_i = self.block.forward_chunkwise(X[:, i*chunk_size:(i+1)*chunk_size, :], r_n_1s, i)
            Y_chunkwise.append(y_n)
            r_n_1s = r_i
            
        Y_chunkwise = torch.concat(Y_chunkwise, dim=1).float()
        
        body_logits = self.body_logits(Y_chunkwise)
        lhand_logits = self.lhand_logits(Y_chunkwise)
        rhand_logits = self.rhand_logits(Y_chunkwise)
        body_softmax_scores = F.softmax(body_logits, dim=2)
        body_latents = torch.argmax(body_softmax_scores, dim=2)
        lhand_softmax_scores = F.softmax(lhand_logits, dim=2)
        lhand_latents = torch.argmax(lhand_softmax_scores, dim=2)
        rhand_softmax_scores = F.softmax(rhand_logits, dim=2)
        rhand_latents = torch.argmax(rhand_softmax_scores, dim=2)

        return body_latents.long(),lhand_latents.long(),rhand_latents.long()
              
    def sample(self,audio,id):
        
        body_latents=torch.zeros((1, audio.shape[2]),dtype=torch.int64, device=audio.device)
        body_latents[:,:1] = torch.randint(low=0,high=2023,size=(1,1))
        lhand_latents=torch.zeros((1, audio.shape[2]),dtype=torch.int64, device=audio.device)
        lhand_latents[:,:1] = torch.randint(low=0,high=2023,size=(1,1))
        rhand_latents=torch.zeros((1, audio.shape[2]),dtype=torch.int64, device=audio.device)
        rhand_latents[:,:1] = torch.randint(low=0,high=2023,size=(1,1))
        
        with torch.no_grad():
            
            for k in range(audio.shape[2]):    
             
              body_logits,lhand_logits,rhand_logits = self.forward(body_latents[:,:k+1],lhand_latents[:,:k+1],rhand_latents[:,:k+1],audio[:,:,:k+1],id)     
                            
              body_probs = F.softmax(body_logits[:,k], dim=-1)
              body_latents[:,k] = body_probs.multinomial(1)  
              lhand_probs = F.softmax(lhand_logits[:,k], dim=-1)
              lhand_latents[:,k] = lhand_probs.multinomial(1)  
              rhand_probs = F.softmax(rhand_logits[:,k], dim=-1)
              rhand_latents[:,k] = rhand_probs.multinomial(1) 
              
        return body_latents,lhand_latents,rhand_latents
        
       