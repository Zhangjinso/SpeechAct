import os
import sys

sys.path.append(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataloaders import torch_data

from trainer.options import parse_args
from trainer.config import load_JsonConfig
from trainer.init_model import init_model

import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import random
import logging
import time
import shutil

class Trainer():
    def __init__(self) -> None:
        parser = parse_args()
        self.args = parser.parse_args()
        self.config = load_JsonConfig(self.args.config_file)
        
        self.device = torch.device(self.args.gpu)
        torch.cuda.set_device(self.device)
        self.setup_seed(self.args.seed)
        self.set_train_dir()

        shutil.copy(self.args.config_file, self.train_dir)

        self.generator = init_model(self.config.Model.model_name, self.args, self.config)
        self.init_dataloader()
        self.start_epoch = 0
        self.global_steps = 0
        if self.args.resume:
            self.resume()
        # self.init_optimizer()

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def set_train_dir(self):
        time_stamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        train_dir = os.path.join(os.getcwd(), self.args.save_dir, os.path.normpath(
            time_stamp + '-' + self.args.exp_name + '-' + self.config.Log.name))
        # train_dir= os.path.join(os.getcwd(), self.args.save_dir, os.path.normpath(time_stamp+'-'+self.args.exp_name+'-'+time.strftime("%H:%M:%S")))
        os.makedirs(train_dir, exist_ok=True)
        log_file=os.path.join(train_dir, 'train.log')

        fmt="%(asctime)s-%(lineno)d-%(message)s"
        logging.basicConfig(
            stream=sys.stdout, level=logging.INFO,format=fmt, datefmt='%m/%d %I:%M:%S %p'
        )
        fh=logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
        self.train_dir = train_dir

    def resume(self):
        print('resume from a previous ckpt')
        ckpt = torch.load(self.args.pretrained_pth)
        self.generator.load_state_dict(ckpt['generator'])
        self.start_epoch = ckpt['epoch']
        self.global_steps = ckpt['global_steps']
        self.generator.global_step = self.global_steps


    def init_dataloader(self):

        data_class = torch_data

        self.train_set = data_class(
                data_root=self.config.Data.data_root,
                speakers=self.args.speakers,
                split='train',
                num_frames=self.config.Data.pose.generate_length,
                aud_feat_win_size=self.config.Data.aud.aud_feat_win_size,
                aud_feat_dim=self.config.Data.aud.aud_feat_dim,
                feat_method=self.config.Data.aud.feat_method,
                context_info=self.config.Data.aud.context_info,
                smplx=True,
                audio_sr=16000,
                expression=self.config.Data.pose.expression,
                config=self.config,
                device = self.device
            )
        
        self.train_set.get_dataset()
        self.train_loader = data.DataLoader(self.train_set.all_dataset,
                                            batch_size=self.config.DataLoader.batch_size, shuffle=True,
                                            num_workers=self.config.DataLoader.num_workers, drop_last=True)

    def init_optimizer(self):
        pass

    def print_func(self, loss_dict, steps):
        info_str = ['global_steps:%d'%(self.global_steps)]
        info_str += ['%s:%.4f'%(key, loss_dict[key]/steps) for key in list(loss_dict.keys())]
        
        f = open (r"vqvae_loss_ret.txt","a",encoding="UTF-8")      
        f.write(str(info_str))
        f.write('\n')
        f.close()
        
        logging.info(','.join(info_str))


    def save_model(self, epoch):
        
        state_dict = {
            'generator': self.generator.state_dict(),
            'epoch': epoch,
            'global_steps': self.global_steps
        }
        save_name = os.path.join(self.train_dir, 'ckpt-%d.pth'%(epoch))
        torch.save(state_dict, save_name)

    def train_epoch(self, epoch):
        epoch_loss_dict = {} 
        epoch_steps = 0
        if 'freeMo' in self.config.Model.model_name:
            for bat in zip(self.trans_loader, self.zero_loader):
                self.global_steps += 1
                epoch_steps += 1
                _, loss_dict = self.generator(bat)
                
                if epoch_loss_dict:#非空
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] += loss_dict[key]
                else:
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] = loss_dict[key]

                if self.global_steps % self.config.Log.print_every == 0:
                    self.print_func(epoch_loss_dict, epoch_steps)
        else:
            # self.config.Model.model_name==smplx_S2G
            for bat in self.train_loader:
                self.global_steps += 1
                epoch_steps += 1
                bat['epoch'] = epoch
                #print('bat.keys():',bat.keys())

                _, loss_dict = self.generator(bat)
                if epoch_loss_dict:#非空
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] += loss_dict[key]
                else:
                    for key in list(loss_dict.keys()):
                        epoch_loss_dict[key] = loss_dict[key]
                if self.global_steps % self.config.Log.print_every == 0:
                    self.print_func(epoch_loss_dict, epoch_steps)

    def train(self):
        logging.info('start_training')
        self.total_loss_dict = {}
        for epoch in range(self.start_epoch, self.config.Train.epochs):
        
            f = open (r"vqvae_loss_ret.txt","a",encoding="UTF-8")      
            f.write('epoch:%d\n'%(epoch))
            f.close()
            
            logging.info('epoch:%d'%(epoch))
            self.train_epoch(epoch)
            # self.generator.scheduler.step()
            # logging.info('learning rate:%d' % (self.generator.scheduler.get_lr()[0]))
            if (epoch+1)%self.config.Log.save_every == 0 or (epoch+1) == 30:
                self.save_model(epoch)
if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()