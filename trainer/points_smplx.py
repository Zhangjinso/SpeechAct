import os
import sys

sys.path.append(os.getcwd())
import torch
from trainer.base import TrainWrapperBaseClass
# from nets.spg.faceformer import Faceformer
from models.points_smplx.points2smplx import Points2Smplx as s2a_points
import torch.nn.functional as F

import torch.optim as optim
import smplx


class TrainWrapper(TrainWrapperBaseClass):
  
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.num_classes = self.config.Data.num_classes
        self.init_params()
        model = smplx.create(r'/mnt/16T/zmj2/PyMAF-X-smplx/data/smpl/SMPLX_NEUTRAL_2020.npz', model_type='smplx',
                      gender='NEUTRAL', use_face_contour=False,
                      num_betas=10,
                      num_expression_coeffs=10,
                      ext='npz').to(self.device)
        self.body = s2a_points(self.each_dim[0],256,self.each_dim[3],self.num_classes).to(self.device)
        self.lhand = s2a_points(self.each_dim[1],64,self.each_dim[4],self.num_classes).to(self.device)
        self.rhand = s2a_points(self.each_dim[2],64,self.each_dim[5],self.num_classes).to(self.device)
        self.discriminator = None

        self.Loss = torch.nn.L1Loss().to(self.device)
        super().__init__(args, config)

    def init_optimizer(self):
        print('using Adam')
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
        
        body_point_dim = 55*3+431*3 -3
        lhand_point_dim = 15*3+25*3
        rhand_point_dim = 15*3+24*3
        body_dim = 165
        lhand = 45
        rhand = 45
        self.each_dim = [body_point_dim,lhand_point_dim,rhand_point_dim,body_dim,lhand,rhand]  

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        down_vertices,joints_xyz = bat['down_vertices'].to(self.device).to(torch.float32), bat['joints_xyz'].to(self.device).to(torch.float32)
        down_vertices_lhand,joints_xyz_lhand = bat['down_vertices_lhand'].to(self.device).to(torch.float32), bat['joints_xyz_lhand'].to(self.device).to(torch.float32)
        down_vertices_rhand,joints_xyz_rhand = bat['down_vertices_rhand'].to(self.device).to(torch.float32), bat['joints_xyz_rhand'].to(self.device).to(torch.float32)
        gt_poses = bat['poses'].to(self.device).to(torch.float32)
        betas = bat['betas'].to(self.device).to(torch.float32)
        id = bat['speaker'].to(self.device)
        id = F.one_hot(id, self.num_classes)
        bs,frame,_,_ = joints_xyz.shape

        gt_points_body = torch.cat([joints_xyz,down_vertices],dim=2).reshape(bs,frame,-1).transpose(1,2)
        gt_points_lhand = torch.cat([joints_xyz_lhand,down_vertices_lhand],dim=2).reshape(bs,frame,-1).transpose(1,2)
        gt_points_rhand = torch.cat([joints_xyz_rhand,down_vertices_rhand],dim=2).reshape(bs,frame,-1).transpose(1,2)
        
        loss = 0
        loss_dict, loss = self.vq_train(gt_points_body[:,3:],gt_poses[:,3:],id,'body', self.body, loss_dict, loss)
        loss_dict, loss = self.vq_train(gt_points_lhand,gt_poses[:,-90:-45],id,'lhand', self.lhand, loss_dict, loss)
        loss_dict, loss = self.vq_train(gt_points_rhand,gt_poses[:,-45:],id,'rhand', self.rhand, loss_dict, loss)

        return total_loss, loss_dict

    def vq_train(self, gt_points, gt_poses,id, name, model, dict, total_loss, pre=None):
        pred_poses  = model(gt_points,id)
        loss, loss_dict = self.get_loss(name, pred_poses=pred_poses, gt_poses=gt_poses)
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
                 pred_poses,
                 gt_poses):
        loss_dict = {}

        MSELoss = self.Loss(pred_poses,gt_poses)
        
        ###velocity_loss
        velocity_loss = self.Loss(pred_poses[:, :,1:] - pred_poses[:, :,:-1],gt_poses[:, :,1:] - gt_poses[:, :,:-1])
     
        gen_loss = MSELoss + velocity_loss
        #print('gen_loss:',gen_loss)
        loss_dict['MSELoss'] = MSELoss       
        loss_dict['velocity_loss'] = velocity_loss
       
        return gen_loss, loss_dict

    def infer_on_audio(self, gt_points):
        
        output = []

       
        self.generator.eval()

        with torch.no_grad():
            mid_poses,pred_poses = self.generator(gt_points)
            pred_poses = pred_poses.cpu().numpy()
        output = pred_poses

        return output
