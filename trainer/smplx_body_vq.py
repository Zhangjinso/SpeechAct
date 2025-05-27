import os
import sys
import torch
sys.path.append(os.getcwd())
from trainer.base import TrainWrapperBaseClass
import numpy as np
import torch.optim as optim
from models.stage1_vqvae.vqvae import TripleGrainVQModel as vq_body

class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.init_params()
        self.num_classes = self.config.Data.num_classes
        self.composition = self.config.Model.composition

        self.body = vq_body(in_channels=self.each_dim[0],feature_ch=256,vae_codebook_size=2048,vae_dim=512,resolution = self.config.Data.pose.generate_length,quant_sample_temperature=0.0).to(self.device)
        self.lhand = vq_body(in_channels=self.each_dim[2],feature_ch=64,vae_codebook_size=2048,vae_dim=128,resolution = self.config.Data.pose.generate_length,quant_sample_temperature=0.0).to(self.device)
        self.rhand = vq_body(in_channels=self.each_dim[3],feature_ch=64,vae_codebook_size=2048,vae_dim=128,resolution = self.config.Data.pose.generate_length,quant_sample_temperature=0.0).to(self.device)

        self.Loss = torch.nn.L1Loss()

        super().__init__(args, config)

    def init_optimizer(self):
        #print('using Adam')
        self.body_optimizer = optim.Adam(
                self.body.parameters(),
                lr=self.config.Train.learning_rate.generator_learning_rate,
                betas=[0.9, 0.999]
            )
        self.lhand_optimizer = optim.Adam(
                self.lhand.parameters(),
                lr=self.config.Train.learning_rate.generator_learning_rate,
                betas=[0.9, 0.999]
            )
        self.rhand_optimizer = optim.Adam(
                self.rhand.parameters(),
                lr=self.config.Train.learning_rate.generator_learning_rate,
                betas=[0.9, 0.999]
            )
    def state_dict(self):
        model_state = {
                'body': self.body.state_dict(),
                'body_optim': self.body_optimizer.state_dict(),
                'lhand': self.lhand.state_dict(),
                'lhand_optim': self.lhand_optimizer.state_dict(),
                'rhand': self.rhand.state_dict(),
                'rhand_optim': self.rhand_optimizer.state_dict(),
 
            }
        return model_state

    def init_params(self):
        point_dim = 55*3+431*3
        body_dim = 165
        lhand = 15*3+25*3
        rhand = 15*3+24*3
        face_dim = 100
        self.each_dim = [point_dim,body_dim,lhand,rhand,face_dim]  
       
    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        down_vertices,joints_xyz = bat['down_vertices'].to(self.device).to(torch.float32), bat['joints_xyz'].to(self.device).to(torch.float32)
        down_vertices_lhand,joints_xyz_lhand = bat['down_vertices_lhand'].to(self.device).to(torch.float32), bat['joints_xyz_lhand'].to(self.device).to(torch.float32)
        down_vertices_rhand,joints_xyz_rhand = bat['down_vertices_rhand'].to(self.device).to(torch.float32), bat['joints_xyz_rhand'].to(self.device).to(torch.float32)
        gt_poses = bat['poses'].to(self.device).to(torch.float32)
       
        bs,frame,_,_ = joints_xyz.shape

        gt_points = torch.cat([joints_xyz,down_vertices],dim=2).reshape(bs,frame,-1).transpose(1,2)
        gt_lhand = torch.cat([joints_xyz_lhand,down_vertices_lhand],dim=2).reshape(bs,frame,-1).transpose(1,2)
        gt_rhand = torch.cat([joints_xyz_rhand,down_vertices_rhand],dim=2).reshape(bs,frame,-1).transpose(1,2)

        loss = 0
        loss_dict, loss = self.vq_train(gt_points,'body', self.body, loss_dict, loss)
        loss_dict, loss = self.vq_train(gt_lhand,'lhand', self.lhand, loss_dict, loss)
        loss_dict, loss = self.vq_train(gt_rhand,'rhand', self.rhand, loss_dict, loss)

        return total_loss, loss_dict

    def vq_train(self, gt, name, model, dict, total_loss, pre=None):
        pred, diff = model(gt)

        loss, loss_dict = self.get_loss(name, pred=pred, gt=gt, e_q_loss=diff, pre=pre)
        if name =='body':
            optimizer_name = 'body_optimizer'
        elif name == 'lhand':
            optimizer_name = 'lhand_optimizer'
        elif name == 'rhand':
            optimizer_name = 'rhand_optimizer'

        optimizer = getattr(self,optimizer_name)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for key in list(loss_dict.keys()):
            dict[name + key] = loss_dict.get(key, 0).item()
        return dict, total_loss

    def get_loss(self,
                 name,
                 pred,
                 gt,
                 e_q_loss,
                 pre=None
                 ):
        loss_dict = {}
        
        ### rec_loss               
        if name == 'body':
            transl_rec_loss = self.Loss(pred,gt) 
        else:
            transl_rec_loss = 0

        rec_loss = self.Loss(pred,gt)  + transl_rec_loss
        
        ###velocity_loss
        velocity_loss = self.Loss(pred[:, :,1:] - pred[:, :,:-1],gt[:, :,1:] - gt[:, :,:-1])
        
        ###acceleration_loss
        acceleration_loss = self.Loss(pred[:, :,2:] + pred[:,:,:-2] - 2 * pred[:, :,1:-1],gt[:, :, 2:] + gt[:, :, :-2] - 2 * gt[:, :,1:-1])
      
        gen_loss = 0.02*e_q_loss + rec_loss + 0.5*velocity_loss + 0.5*acceleration_loss 
        
        
        loss_dict['rec_loss'] = rec_loss
        loss_dict['velocity_loss'] = velocity_loss
        loss_dict['acceleration_loss'] = acceleration_loss

        return gen_loss, loss_dict

