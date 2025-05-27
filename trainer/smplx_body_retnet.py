import os
import sys

import torch
from torch.optim.lr_scheduler import StepLR
import random
sys.path.append(os.getcwd())

from trainer.base import TrainWrapperBaseClass
from models.stage2_retnet.s2a_ret import Audio2Motion_RetNet
from models.audio_encoder.encoder import AudioEncoder
from models.utils import get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from models.stage1_vqvae.vqvae import TripleGrainVQModel as vq_body
#from smooth.smooth_pose import smooth_pose
import pickle
from models.points_smplx.points2smplx import Points2Smplx
from models.modules import dequeue_data, queue_data

def make_queue(queue,mapping):
    queue = queue_data(queue, mapping)
    queue = dequeue_data(queue, K=1024)
    return queue  

def mask_latents(latents,mask_ratio):
    mask = torch.bernoulli(mask_ratio * torch.ones(latents.shape, device=latents.device))
    mask = mask.round().to(dtype=torch.int64)
    indices = torch.zeros_like(latents)  
    indices = mask*latents+(1-mask)*indices
    return indices


def calc_contrastive_loss(query, key, queue, temp=0.07):
    N = query.shape[0]
    K = queue.shape[0]

    zeros = torch.zeros(N, dtype=torch.long, device=query.device)
    key = key.detach()
    logit_pos = torch.bmm(query.view(N, 1, -1), key.view(N, -1, 1))
    logit_neg = torch.mm(query.view(N, -1), queue.t().view(-1, K))

    logit = torch.cat([logit_pos.view(N, 1), logit_neg], dim=1)

    loss = F.cross_entropy(logit / temp, zeros)

    return loss


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0
        self.epoch = 0
        self.init_params()
        self.num_classes = self.config.Data.num_classes
        self.audio = True
        self.contrastive_weight = 0.0002
        self.audioencoder = AudioEncoder(in_dim=65, num_hiddens=256, num_residual_layers=2, num_residual_hiddens=256).to(self.device)
        
        self.generator = Audio2Motion_RetNet(num_vq=2048, embed_dim=256, clip_dim=256, block_size=64, num_layers=16, n_head=16, drop_out_rate=0.1, fc_rate=3,n_classes = self.num_classes).to(self.device)
        
        self.body = vq_body(in_channels=self.each_dim[0],feature_ch=256,vae_codebook_size=2048,vae_dim=512,resolution = self.config.Data.pose.generate_length,quant_sample_temperature=0.0).to(self.device)
        self.lhand = vq_body(in_channels=self.each_dim[1],feature_ch=64,vae_codebook_size=2048,vae_dim=128,resolution = self.config.Data.pose.generate_length,quant_sample_temperature=0.0).to(self.device)
        self.rhand = vq_body(in_channels=self.each_dim[2],feature_ch=64,vae_codebook_size=2048,vae_dim=128,resolution = self.config.Data.pose.generate_length,quant_sample_temperature=0.0).to(self.device)
        
        model_path = self.config.Model.vq_path
        model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))

        self.body.load_state_dict(model_ckpt['generator']['body'])
        self.lhand.load_state_dict(model_ckpt['generator']['lhand'])
        self.rhand.load_state_dict(model_ckpt['generator']['rhand'])
        
        self.body_smplx = Points2Smplx(self.each_dim[0]-3,256,self.each_dim[3],self.num_classes).to(self.device)
        self.lhand_smplx = Points2Smplx(self.each_dim[1],64,self.each_dim[4],self.num_classes).to(self.device)
        self.rhand_smplx = Points2Smplx(self.each_dim[2],64,self.each_dim[5],self.num_classes).to(self.device)
        
        model_ckpt_v2 = torch.load(self.config.Model.p2s_path, map_location=torch.device('cpu'))
        #model_ckpt_v2 = torch.load("./experiments/talkshow_2024-05-02-s2a-points/ckpt-99.pth", map_location=torch.device('cpu'))
        self.body_smplx.load_state_dict(model_ckpt_v2['generator']['body'])
        self.lhand_smplx.load_state_dict(model_ckpt_v2['generator']['lhand'])
        self.rhand_smplx.load_state_dict(model_ckpt_v2['generator']['rhand'])
        
        self.Loss = torch.nn.L1Loss() 
        self.body_queue = torch.zeros((0, 512), dtype=torch.float).cuda() 
        self.lhand_queue = torch.zeros((0, 128), dtype=torch.float).cuda() 
        self.rhand_queue = torch.zeros((0, 128), dtype=torch.float).cuda()               
        super().__init__(args, config)
    
    def init_optimizer(self):

        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.999]
        )
        if self.audioencoder is not None:
            opt = self.config.Model.AudioOpt
            if opt == 'Adam':
                self.audioencoder_optimizer = optim.Adam(
                    self.audioencoder.parameters(),
                    lr=self.config.Train.learning_rate.generator_learning_rate,
                    betas=[0.9, 0.999]
                )
            else:
                print('using SGD')
                self.audioencoder_optimizer = optim.SGD(
                filter(lambda p: p.requires_grad,self.audioencoder.parameters()),
                lr=self.config.Train.learning_rate.generator_learning_rate*10,
                momentum=0.9,
                nesterov=False)
                             
    def state_dict(self):
        model_state = {
            'generator': self.generator.state_dict(),
            'generator_optim': self.generator_optimizer.state_dict(),
            'audioencoder': self.audioencoder.state_dict(),
            'audioencoder_optim': self.audioencoder_optimizer.state_dict(),
        }
        return model_state

    def load_state_dict(self, state_dict):

        from collections import OrderedDict
        new_state_dict = OrderedDict()  # create new OrderedDict that does not contain `module.`
        for k, v in state_dict.items():
            sub_dict = OrderedDict()
            if v is not None:
                for k1, v1 in v.items():
                    name = k1.replace('module.', '')
                    sub_dict[name] = v1
            new_state_dict[k] = sub_dict
        state_dict = new_state_dict
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        
        if 'generator_optim' in state_dict and self.generator_optimizer is not None:
            self.generator_optimizer.load_state_dict(state_dict['generator_optim'])

       
        if 'audioencoder' in state_dict:
            self.audioencoder.load_state_dict(state_dict['audioencoder'])
        if 'audioencoder_optimizer' in state_dict:
            self.audioencoder_optimizer.load_state_dict(state_dict['audioencoder_optimizer'])
     
    def init_params(self):
        
        body_point_dim = 55*3+431*3
        body_dim = 165
        lhand_point_dim = 15*3+25*3
        lhand_dim = 45
        rhand_point_dim = 15*3+24*3
        rhand_dim = 45
        face_dim = 100
        
        self.each_dim = [body_point_dim,lhand_point_dim,rhand_point_dim,body_dim,lhand_dim,rhand_dim,face_dim]  
        #print('self.each_dim:',self.each_dim) 

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}
        epoch = bat['epoch']
        aud = bat['aud_feat'].to(self.device).to(torch.float32)
        down_vertices,joints_xyz = bat['down_vertices'].to(self.device).to(torch.float32), bat['joints_xyz'].to(self.device).to(torch.float32)
        down_vertices_lhand,joints_xyz_lhand = bat['down_vertices_lhand'].to(self.device).to(torch.float32), bat['joints_xyz_lhand'].to(self.device).to(torch.float32)
        down_vertices_rhand,joints_xyz_rhand = bat['down_vertices_rhand'].to(self.device).to(torch.float32), bat['joints_xyz_rhand'].to(self.device).to(torch.float32)
        gt_poses = bat['poses'].to(self.device).to(torch.float32)
        
        betas = bat['betas'].to(self.device).to(torch.float32)
        
        bs,frame,_,_ = joints_xyz.shape
        
        gt_points = torch.cat([joints_xyz,down_vertices],dim=2).reshape(bs,frame,-1).transpose(1,2)  
        gt_lhand = torch.cat([joints_xyz_lhand,down_vertices_lhand],dim=2).reshape(bs,frame,-1).transpose(1,2)
        gt_rhand = torch.cat([joints_xyz_rhand,down_vertices_rhand],dim=2).reshape(bs,frame,-1).transpose(1,2)     
        id = bat['speaker'].to(self.device)
        one_hot_id = F.one_hot(id, self.num_classes)

        with torch.no_grad():
            self.body.eval()
            self.lhand.eval()
            self.rhand.eval()
            body_gt_fea,_,body_latents = self.body.encode(gt_points)
            lhand_gt_fea,_,lhand_latents = self.lhand.encode(gt_lhand)
            rhand_gt_fea,_,rhand_latents = self.rhand.encode(gt_rhand)
  
        audio = self.audioencoder(aud.transpose(1,2), frame_num=body_latents.shape[1]*4)  ### torch.Size([64, 256, 32])
        #text = self.textencoder(text_feat_seq.transpose(1,2), frame_num=latents.shape[1]*4)
        body_latents1 = mask_latents(body_latents,0.5)
        lhand_latents1 = mask_latents(lhand_latents,0.5)
        rhand_latents1 = mask_latents(rhand_latents,0.5)
        
        body_logits,lhand_logits,rhand_logits = self.generator(body_latents1.long(),lhand_latents1.long(),rhand_latents1.long(),audio,id)
            
        self.generator_optimizer.zero_grad()
        self.audioencoder_optimizer.zero_grad()
                
        ce_loss = F.cross_entropy(body_logits.view(-1, body_logits.shape[-1]), body_latents.view(-1))\
                  + F.cross_entropy(lhand_logits.view(-1, lhand_logits.shape[-1]), lhand_latents.view(-1))\
                  + F.cross_entropy(rhand_logits.view(-1, rhand_logits.shape[-1]), rhand_latents.view(-1))
        loss = ce_loss 
        
        #print('loss:',loss)
        loss.backward()

        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.config.Train.max_gradient_norm)

        if torch.isnan(grad).sum() > 0:
            print('fuck')

        loss_dict['grad'] = grad.item()
        loss_dict['loss'] = loss
        
        self.generator_optimizer.step()
        self.audioencoder_optimizer.step()

        return total_loss, loss_dict
    
    def infer_on_audio(self, aud_fn, id=None, fps=30, sr=16000, am=None, am_sr=None, frame=0):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        import time
        #start_time = time.time()
        output = []

        assert self.args.infer, "train mode"
        self.generator.eval()
        self.body.eval()
        self.lhand.eval()
        self.rhand.eval()

        aud_feat = get_mfcc_ta(aud_fn, sr=sr, fps=fps, smlpx=True, type='mfcc', am=am)
        
    
        aud_feat = aud_feat.transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(1, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)
        one_hot_id = F.one_hot(id, self.num_classes)

        with torch.no_grad():
            audio = self.audioencoder(aud_feat, frame_num=frame)         
            ###body_latents,lhand_latents,rhand_latents = self.generator.sample(audio,id)
            body_latents,lhand_latents,rhand_latents = self.generator.sample_chunkwise(audio,id)
           
            _,pred_points = self.body.decode_code(body_latents)
            _,pred_lhand_points = self.lhand.decode_code(lhand_latents)
            _,pred_rhand_points = self.rhand.decode_code(rhand_latents)
            
            bs,_,frame = pred_points.shape
            #print('pred_points:',pred_points.shape)
            
            pred_points[:,1,1:] = pred_points[:,1,1:] - pred_points[:,1,:1]
            pred_points[:,:3,0] = torch.tensor([0,0,0]).type(torch.cuda.FloatTensor)
            for i in range(1, frame):
                pred_points[:,[0,2],i] = pred_points[:,[0,2],i] + pred_points[:,[0,2],i-1]
            root = pred_points[:,:3,:].cpu()

            pred_smplx = self.body_smplx(pred_points[:,3:,:],one_hot_id)
            pred_lhand = self.lhand_smplx(pred_lhand_points,one_hot_id)
            pred_rhand = self.rhand_smplx(pred_rhand_points,one_hot_id)
            pred_smplx[:,-90:-45] = pred_lhand  
            pred_smplx[:,-45:] = pred_rhand
            pred_smplx = torch.cat([root,torch.tensor(pred_smplx).cpu()],dim=1).transpose(1,2)
            
        return pred_smplx