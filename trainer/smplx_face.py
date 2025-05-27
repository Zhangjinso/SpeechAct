import os
import sys
import torch
sys.path.append(os.getcwd())

from trainer.base import TrainWrapperBaseClass
# from nets.spg.faceformer import Faceformer
from models.face.s2a_face import Generator as s2a_face
from models.utils import get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


        
class TrainWrapper(TrainWrapperBaseClass):

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.init_params()
        self.num_classes = self.config.Data.num_classes

        self.generator = s2a_face(
            n_poses=self.config.Data.pose.generate_length,
            each_dim=self.each_dim,
            training=not self.args.infer,
            device=self.device,
            identity=True,
            num_classes=self.num_classes,
        ).to(self.device)

        # self.generator = Faceformer().to(self.device)
        self.discriminator = None
        self.am = None
        self.Loss = torch.nn.L1Loss()
        super().__init__(args, config)

    def init_optimizer(self):
        self.generator_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad,self.generator.parameters()),
            lr=0.001,
            momentum=0.9,
            nesterov=False,
        )

    def init_params(self):

        jaw_dim = 3

        face_dim = 100
        
        self.each_dim = [jaw_dim, face_dim]

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        id = bat['speaker'].to(self.device)
        id = F.one_hot(id, self.num_classes)
       
        aud = aud.permute(0, 2, 1)
        gt_poses = poses.permute(0, 2, 1)
        
        if self.expression:
            expression = bat['expression'].to(self.device).to(torch.float32)
            gt_poses = torch.cat([gt_poses, expression.permute(0, 2, 1)], dim=2)
        pred_poses, _ = self.generator(
            aud,
            gt_poses,
            id,
        )
        #print('pred_poses:',pred_poses.shape)
        G_loss, G_loss_dict = self.get_loss(
            pred_poses=pred_poses,
            gt_poses=torch.cat([gt_poses[:, :, 3:6],gt_poses[:, :, -100:]],dim=-1))

        self.generator_optimizer.zero_grad()
        G_loss.backward()
        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['grad'] = grad.item()
        self.generator_optimizer.step()

        for key in list(G_loss_dict.keys()):
            loss_dict[key] = G_loss_dict.get(key, 0).item()

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses):
        loss_dict = {}

        rec_loss = self.Loss(pred_poses,gt_poses) 
        velocity_loss = self.Loss(pred_poses[:, 1:] - pred_poses[:, :-1],gt_poses[:, 1:] - gt_poses[:, :-1])
        gen_loss = rec_loss + velocity_loss

        loss_dict['rec_loss'] = rec_loss
        if self.expression:
            loss_dict['velocity_loss'] = velocity_loss
        
        return gen_loss, loss_dict

    def infer_on_audio(self, aud_fn, id=None, frame=None, am=None, am_sr=16000, **kwargs):
       
        self.generator.eval()

        aud_feat = get_mfcc_ta(aud_fn, am=am, am_sr=am_sr, fps=30, encoder_choice='faceformer')
        aud_feat = aud_feat[np.newaxis, ...].repeat(1, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.generator.device).transpose(1, 2)
        
        frame = aud_feat.shape[2]*30//16000
       
        id = F.one_hot(id, self.num_classes).unsqueeze(0).to(self.generator.device)
        
        with torch.no_grad():
            if aud_feat.shape[2] > 1000000:
                pred_poses = []
                aud_feats = torch.chunk(aud_feat, 10, 2)
                for i in aud_feats:
                    #print(i.shape)
                    pred_pose = self.generator(i, None, id, time_steps=frame//10)[0]
                    pred_poses.append(pred_pose.cpu())
                    #print(pred_pose.shape)
                pred_poses = torch.cat(pred_poses, 1).numpy()
            else:
                pred_poses = self.generator(aud_feat, None, id, time_steps=frame)[0]
                pred_poses = pred_poses.cpu().numpy()
        

        return pred_poses
