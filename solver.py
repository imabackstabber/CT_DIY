import os
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from loader import *

from networks import *
from measure import compute_measure
from misc import AverageMeter
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

class Solver(object):
    def __init__(self, args, train_data_loader, test_data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.val_iters = args.val_iters # val every given iters
        self.result_fig = args.result_fig
        self.patch_size = args.patch_size
        self.batch_size = args.batch_size

        # assert self.val_iters % (self.batch_size * len(self.train_data_loader)) == 0, \
        #         'val iters must divide length of self.train_data_loader, ' \
        #         'got loader {} and val_iters {}'.format(len(self.train_data_loader) * self.batch_size, self.val_iters)

        self.patch_training = args.patch_training
        self.norm = args.norm

        if self.norm:
            self.trunc_min = -1
            self.trunc_max = 1
        
        self.lr = args.lr
        self.model_name = args.model

        # create OmegaDict
        self.conf = OmegaConf.load(args.yaml_path)
        for key,value in vars(args).items():
            OmegaConf.update(self.conf, key, value)

        
        if self.model_name == 'REDCNN':
            self.fig_save_path = os.path.join(self.save_path, self.model_name, 'fig')
            self.model = RED_CNN()
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.model.parameters(), self.lr)
            self.model.to(self.device)
            if (self.multi_gpu) and (torch.cuda.device_count() > 1):
                logger.info('Use {} GPUs'.format(torch.cuda.device_count()))
                self.model = nn.DataParallel(self.model)
            
        elif self.model_name in ['cncl_unet', 'cncl_attn']:
            # alternatively training gan
            self.alter_training = args.alter_gan
            if self.conf.attn_mode == 'base':
                self.model_name = self.model_name + '_base'
            elif self.conf.attn_mode == 'ca':
                self.model_name = self.model_name + '_ca'
                if self.conf.noise_mode == 'sk':
                    self.model_name = self.model_name + '_ca_sk'
            elif self.conf.attn_mode == 'sca':
                self.model_name = self.model_name + '_sca'
            else:
                raise NotImplementedError('CNCL mode: {} Not Implemented.'.format(self.conf.cncl_mode))

            # hack: add diffrent norm + act
            self.model_name += '_{}_{}'.format(self.conf.norm_mode, self.conf.act_mode)

            if self.model_name.startswith('cncl_unet'):
                self.generator = CNCL_unet(
                    noise_encoder=self.conf.noise_mode, 
                    content_encoder=self.conf.content_mode, 
                    attn_mode=self.conf.attn_mode,
                    norm_mode=self.conf.norm_mode,
                    act_mode=self.conf.act_mode
                    )
            elif self.model_name.startswith('cncl_attn'):
                # update layer info
                self.mdta_num = self.conf.mdta_num
                self.cross_num = self.conf.cross_num
                self.model_name += '_{}_{}'.format(self.mdta_num, self.cross_num)
                self.generator = CNCL_attn(
                    noise_encoder=self.conf.noise_mode, 
                    content_encoder=self.conf.content_mode, 
                    attn_mode=self.conf.attn_mode,
                    norm_mode=self.conf.norm_mode,
                    act_mode=self.conf.act_mode,
                    mdta_layer_num = self.conf.mdta_num,
                    cross_layer_num = self.conf.cross_num
                )
            self.gan = PatchLSGAN()

            self.generator.to(self.device)
            self.gan.to(self.device)
            
            if self.alter_training:
                self.discriminator_iters = args.discriminator_iters
                self.opt_g = optim.Adam(self.generator.parameters(), self.lr)
                self.opt_d = optim.Adam(self.gan.parameters(), self.lr)
            else:
                self.discriminator_iters = 1
                self.generator_lr = args.generator_lr
                self.discriminator_lr = args.discriminator_lr
                self.opt_g = optim.Adam(self.generator.parameters(), self.generator_lr)
                self.opt_d = optim.Adam(self.gan.parameters(), self.discriminator_lr)
            self.fai = 100
            self.lamda = 1
            self.downsample_rate = (16,16)
            self.criterion_content = nn.L1Loss()
            self.criterion_noise = nn.L1Loss()
            self.criterion_gan = nn.MSELoss() # for LSGAN
            self.base_loss_weight = 1.0 # as default
            if 'loss' in self.conf:
                for _loss in self.conf.loss:
                    loss_type, loss_weight = _loss.type, _loss.weight
                    if loss_type == 'base':
                        self.base_loss_weight = loss_weight # override default
                    elif loss_type == 'texture':
                        self.texture_loss_weight = loss_weight
                    elif loss_type == 'perceptual':
                        self.perceptual_loss_weight = loss_weight

            # hack: add diffrent loss name
            self.model_name += '{}{}'.format('_texture_{}'.format(self.texture_loss_weight) if hasattr(self, 'texture_loss_weight') else '',\
                                            '_perceptual_{}'.format(self.perceptual_loss_weight) if hasattr(self, 'perceptual_loss_weight') else '')

            # hack here for cncl
            self.fig_save_path = os.path.join(self.save_path, self.model_name + '_generator', 'fig')
            
            if hasattr(self, 'texture_loss_weight') or hasattr(self, 'perceptual_loss_weight'):
                self.feat_extractor = feat_extractor()

            if (self.multi_gpu) and (torch.cuda.device_count() > 1):
                logger.info('Use {} GPUs'.format(torch.cuda.device_count()))
                self.generator = nn.DataParallel(self.generator)
                self.gan = nn.DataParallel(self.gan)

        else:
            raise NotImplementedError('Architecture {} Not Implemented.'.format(self.model_name))


        # create tensorboard logger
        self.tensorboard_path = str(os.path.join(args.log_dir, self.model_name)) # make it to different arch
        self.writer = SummaryWriter(self.tensorboard_path)
        self.log_path = str(os.path.join(self.tensorboard_path, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))))
        # create loguru handler
        logger.add(self.log_path + '/record.log', 
                backtrace = True, diagnose= True,  format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

        OmegaConf.save(self.conf, self.log_path + '/setting.yaml')
        logger.info(self.conf)

        self.transform = args.transform

        if self.transform:
            # flip and rotate 90
            self.preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([
                    transforms.RandomRotation(90)
                ])
            ])
            logger.debug('USING TRANFORM: Flip and Rotate')
        
    def save_model(self, model, iter_, name = None):
        model_name = name if name else self.model_name
        dir_path = os.path.join(self.save_path,model_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = os.path.join(dir_path, '{}_{}iter_{}_{}.ckpt'.format(model_name , iter_, 
                                                                'patch' if self.patch_training else '', 
                                                                'norm' if self.norm else ''))
        torch.save(model.state_dict(), f)

    def load_model(self, model, iter_, name = None):
        model_name = name if name else self.model_name
        dir_path = os.path.join(self.save_path,model_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = os.path.join(dir_path, '{}_{}iter_{}_{}.ckpt'.format(model_name , iter_, 
                                                                'patch' if self.patch_training else '', 
                                                                'norm' if self.norm else ''))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f).items():
                n = k[7:]
                state_d[n] = v
            model.load_state_dict(state_d)
        else:
            model.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.1
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # cncl, not sure
            for param_group in self.opt_g.param_groups:
                param_group['lr'] = lr
            for param_group in self.opt_d.param_groups:
                param_group['lr'] = lr
            


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)


        if not os.path.exists(self.fig_save_path):
            os.makedirs(self.fig_save_path)
        f.savefig(os.path.join(self.fig_save_path, 'result_{}.png'.format(fig_name)))
        plt.close()

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def transform_before_train(self, img, target):
        if self.transform:
            return self.preprocess(img), self.preprocess(target)
        else:
            return img, target

    def train_redcnn(self):

        total_iters = 0
        param = self.get_parameter_number(self.model)
        logger.info('param={}'.format(param.items()))

        start_time = time.time()
        # add per_iter time
        end = time.time()
        batch_time = AverageMeter()

        tot_iters = (len(self.train_data_loader) * self.num_epochs) # len(dataloader) will return len(dataset) // batch_size

        for epoch in range(0, self.num_epochs):
            for iter_, (x, y) in enumerate(self.train_data_loader):
                self.model.train(True)
                total_iters += 1

                # x, y: [self.batch_size, 512, 512] -> [self.batch_size, self.patch_n, self.patch_size, self.patch_size]
                if self.patch_training: # patch training
                    x = x.reshape(-1, 1, self.patch_size, self.patch_size).float().to(self.device)
                    y = y.reshape(-1, 1, self.patch_size, self.patch_size).float().to(self.device)
                else:
                    x = x.unsqueeze(1).float().to(self.device)
                    y = y.unsqueeze(1).float().to(self.device)

                # add transform
                x, y = self.transform_before_train(x ,y)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # add per_iter time
                batch_time.update(time.time() - end)
                end = time.time()

                # to tensorboard
                self.writer.add_scalar('loss', loss.item(), total_iters)

                # logging
                if total_iters % self.print_iters == 0:
                    eta_seconds = batch_time.avg * (tot_iters - total_iters)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    logger.debug("ETA:[{}], STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(eta_string, total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.train_data_loader), loss.item(), 
                                                                                                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()

                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(self.model, total_iters)

                # val model
                # if total_iters % self.val_iters == 0:
                #     self.val_redcnn(total_iters // self.val_iters)

    def train_cncl(self):
        total_iters = 0
        param_g = self.get_parameter_number(self.generator)
        param_d = self.get_parameter_number(self.gan)
        logger.info('param_g={}'.format(param_g.items()))
        logger.info('param_d={}'.format(param_d.items()))

        start_time = time.time()
        # add per_iter time
        end = time.time()
        batch_time = AverageMeter()

        tot_iters = (len(self.train_data_loader) * self.num_epochs) # len(dataloader) will return len(dataset) // batch_size

        for epoch in range(0, self.num_epochs):
            for iter_, (img, target) in enumerate(self.train_data_loader):
                self.generator.train()
                self.gan.train()
                total_iters += 1

                # x, y: [self.batch_size, 512, 512] -> [self.batch_size, self.patch_n, self.patch_size, self.patch_size]
                if self.patch_training: # patch training
                    img = img.reshape(-1, 1, self.patch_size, self.patch_size).float().to(self.device)
                    target = target.reshape(-1, 1, self.patch_size, self.patch_size).float().to(self.device)
                else:
                    img = img.unsqueeze(1).float().to(self.device)
                    target = target.unsqueeze(1).float().to(self.device)

                # add transform
                img, target = self.transform_before_train(img ,target)

                B,_,H,W = img.shape

                noise = img - target # get ground truth noise

                valid = torch.ones(B,1, H // self.downsample_rate[0], W // self.downsample_rate[1], requires_grad= False).float().to(self.device)
                fake = torch.zeros(B,1, H // self.downsample_rate[0], W // self.downsample_rate[1], requires_grad= False).float().to(self.device)
                

                # generator loss
                res = self.generator(img)
                pred_noise, pred_content, pred_fusion = res['pred_noise'], res['pred_content'], res['pred_fusion']
                pred_fake = self.gan(pred_fusion, img, pred_noise)


                loss_gan_g = self.criterion_gan(pred_fake, valid)
                loss_noise_g = self.criterion_noise(pred_noise, noise)
                loss_content_g = self.criterion_content(pred_fusion, target)
                loss_l1 = loss_content_g + self.lamda * loss_noise_g
                loss_g = loss_gan_g + self.fai * loss_l1
                
                if hasattr(self, 'texture_loss_weight'):
                    loss_g += self.feat_extractor.texture_loss(pred_fusion, target) * self.texture_loss_weight
                if hasattr(self, 'perceptual_loss_weight'):
                    loss_g += self.feat_extractor.perceptual_loss(pred_fusion, target) * self.perceptual_loss_weight
                
                self.opt_g.zero_grad()
                loss_g.backward()
                self.opt_g.step()
                self.writer.add_scalar('loss_g', loss_g.item(), total_iters)

                # discriminator loss

                if total_iters % self.discriminator_iters == 0:
                    # real loss
                    pred_real = self.gan(target, img, noise)
                    loss_real = self.criterion_gan(pred_real, valid)
                    # fake loss
                    pred_fake = self.gan(pred_fusion.detach(), img, pred_noise.detach()) # detach from generator
                    loss_fake = self.criterion_gan(pred_fake, fake)
                    # total loss
                    loss_d = 0.5 * (loss_real + loss_fake)
                    self.opt_d.zero_grad()
                    loss_d.backward()
                    self.opt_d.step()
                    self.writer.add_scalar('loss_d', loss_d.item(), total_iters)

                # add per_iter time
                batch_time.update(time.time() - end)
                end = time.time()

                # logging
                if total_iters % self.print_iters == 0:
                    eta_seconds = batch_time.avg * (tot_iters - total_iters)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    logger.debug('ETA:[{}], STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \n'
                                'LOSS_G: {:.8f}, LOSS_D: {:.8f}, TIME: {:.1f}s'.format(eta_string, total_iters, epoch, 
                                                                    self.num_epochs, iter_+1, 
                                                                    len(self.train_data_loader), 
                                                                    loss_g.item(),
                                                                    loss_d.item(),
                                                                    time.time() - start_time))
                # learning rate decay (not sure whether to use in gan)
                # if total_iters % self.decay_iters == 0:
                #     self.lr_decay()

                # save model
                if total_iters % self.save_iters == 0:
                    self.save_model(self.generator, total_iters, name='{}_generator'.format(self.model_name))
                    self.save_model(self.gan, total_iters, name='{}_gan'.format(self.model_name))

                # val model
                # if total_iters % self.val_iters == 0:
                #     self.val_cncl(total_iters // self.val_iters)

    def train(self):
        if self.model_name.startswith('REDCNN'):
            self.train_redcnn()
        elif self.model_name.startswith('cncl'):
            self.train_cncl()
        else:
            logger.exception('Architecture {} Not Implemented.'.format(self.model_name))
            raise NotImplementedError('Architecture {} Not Implemented.'.format(self.model_name))

    def val_cncl(self, val_iter):
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []


        self.generator.eval().to(self.device)
        self.gan.eval().to(self.device)

        with torch.no_grad():

            for i, (x, y) in enumerate(self.test_data_loader):

                shape_ = x.shape[-1]
                # sync with training
                
                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)

                pred_content, pred_noise, pred = self.generator(x)['pred_fusion']
                # x = self.trunc(self.denormalize_(x.cpu().detach()))
                # y = self.trunc(self.denormalize_(y.cpu().detach()))
                
                # for fusion visualization
                x = self.trunc(self.denormalize_(x.cpu().detach()))
                y = self.trunc(self.denormalize_((y - pred).cpu().detach()))
                pred = self.trunc(self.denormalize_((y - (x - pred_noise)).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_psnr_avg1.append(ori_psnr_avg / len(self.test_data_loader))
                ori_ssim_avg += original_result[1]
                ori_ssim_avg1.append(ori_ssim_avg / len(self.test_data_loader))
                ori_rmse_avg += original_result[2]
                ori_rmse_avg1.append(ori_rmse_avg / len(self.test_data_loader))
                pred_psnr_avg += pred_result[0]
                pred_psnr_avg1.append(pred_psnr_avg / len(self.test_data_loader))
                pred_ssim_avg += pred_result[1]
                pred_ssim_avg1.append(pred_ssim_avg / len(self.test_data_loader))
                pred_rmse_avg += pred_result[2]
                pred_rmse_avg1.append(pred_rmse_avg / len(self.test_data_loader))

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

            logger.debug('Original\tPSNR avg: {:.4f} , SSIM avg: {:.4f} , RMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.test_data_loader), ori_ssim_avg / len(self.test_data_loader),
                ori_rmse_avg / len(self.test_data_loader)))
            logger.debug('After learning\tPSNR avg: {:.4f} , SSIM avg: {:.4f} , RMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.test_data_loader), pred_ssim_avg / len(self.test_data_loader),
                pred_rmse_avg / len(self.test_data_loader)))
            self.writer.add_scalar('psnr', pred_psnr_avg, val_iter)
            self.writer.add_scalar('ssim', pred_ssim_avg, val_iter)
            self.writer.add_scalar('rmse', pred_rmse_avg, val_iter)

    def test_cncl(self):
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []


        self.generator.eval().to(self.device)
        self.gan.eval().to(self.device)

        with torch.no_grad():

            for i, (x, y) in enumerate(self.test_data_loader):

                shape_ = x.shape[-1]
                # sync with training

                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)

                out= self.generator(x)
                pred_content, pred_noise, pred  = out['pred_content'], out['pred_noise'], out['pred_fusion']

                # for fusion visualization
                y_pred = self.trunc(self.denormalize_((pred_content).view(shape_, shape_).cpu().detach()))
                y_noise = self.trunc(self.denormalize_((y - (x - pred_noise)).view(shape_, shape_).cpu().detach()))
                y_fusion = self.trunc(self.denormalize_((y - pred).view(shape_, shape_).cpu().detach()))

                # x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                # y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                # pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_psnr_avg1.append(ori_psnr_avg / len(self.test_data_loader))
                ori_ssim_avg += original_result[1]
                ori_ssim_avg1.append(ori_ssim_avg / len(self.test_data_loader))
                ori_rmse_avg += original_result[2]
                ori_rmse_avg1.append(ori_rmse_avg / len(self.test_data_loader))
                pred_psnr_avg += pred_result[0]
                pred_psnr_avg1.append(pred_psnr_avg / len(self.test_data_loader))
                pred_ssim_avg += pred_result[1]
                pred_ssim_avg1.append(pred_ssim_avg / len(self.test_data_loader))
                pred_rmse_avg += pred_result[2]
                pred_rmse_avg1.append(pred_rmse_avg / len(self.test_data_loader))

                # save result figure
                if self.result_fig:
                    # self.save_fig(x, y, pred, i, original_result, pred_result)
                    self.save_fig(y_pred, y_noise, y_fusion, i, original_result, pred_result)

            logger.debug('Original\tPSNR avg: {:.4f} , SSIM avg: {:.4f} , RMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.test_data_loader), ori_ssim_avg / len(self.test_data_loader),
                ori_rmse_avg / len(self.test_data_loader)))
            logger.debug('After learning\tPSNR avg: {:.4f} , SSIM avg: {:.4f} , RMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.test_data_loader), pred_ssim_avg / len(self.test_data_loader),
                pred_rmse_avg / len(self.test_data_loader)))

    def val_redcnn(self, val_iter):
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []


        self.model.eval()

        with torch.no_grad():

            for i, (x, y) in enumerate(self.test_data_loader):

                shape_ = x.shape[-1]
                # sync with training
                
                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)

                pred = self.model(x)
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_psnr_avg1.append(ori_psnr_avg / len(self.test_data_loader))
                ori_ssim_avg += original_result[1]
                ori_ssim_avg1.append(ori_ssim_avg / len(self.test_data_loader))
                ori_rmse_avg += original_result[2]
                ori_rmse_avg1.append(ori_rmse_avg / len(self.test_data_loader))
                pred_psnr_avg += pred_result[0]
                pred_psnr_avg1.append(pred_psnr_avg / len(self.test_data_loader))
                pred_ssim_avg += pred_result[1]
                pred_ssim_avg1.append(pred_ssim_avg / len(self.test_data_loader))
                pred_rmse_avg += pred_result[2]
                pred_rmse_avg1.append(pred_rmse_avg / len(self.test_data_loader))

                # save result figure
                if self.result_fig:
                    self.save_fig(x.squeeze(), y.squeeze(), pred.squeeze(), i, original_result, pred_result)

            logger.debug('Original\tPSNR avg: {:.4f} , SSIM avg: {:.4f} , RMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.test_data_loader), ori_ssim_avg / len(self.test_data_loader),
                ori_rmse_avg / len(self.test_data_loader)))
            logger.debug('After learning\tPSNR avg: {:.4f} , SSIM avg: {:.4f} , RMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.test_data_loader), pred_ssim_avg / len(self.test_data_loader),
                pred_rmse_avg / len(self.test_data_loader)))
            self.writer.add_scalar('psnr', pred_psnr_avg, val_iter)
            self.writer.add_scalar('ssim', pred_ssim_avg, val_iter)
            self.writer.add_scalar('rmse', pred_rmse_avg, val_iter)

    def test(self):
        # del self.model
        # load
        if self.model_name == 'REDCNN' :
            self.model = RED_CNN().to(self.device)
            self.load_model(self.model, self.test_iters)
            self.model.eval()
        elif self.model_name.startswith('cncl_unet'):
            self.generator = CNCL_unet(
                noise_encoder=self.conf.noise_mode, 
                content_encoder=self.conf.content_mode, 
                attn_mode=self.conf.attn_mode,
                norm_mode=self.conf.norm_mode,
                act_mode=self.conf.act_mode
                )
            self.load_model(self.generator, self.test_iters, self.model_name + '_generator') # hack: append '_generator'
            self.generator.eval()
            self.test_cncl()
            return 
        else:
            logger.exception('Architecture {} Not Implemented.'.format(self.model_name))
            raise NotImplementedError('Architecture {} Not Implemented.'.format(self.model_name))

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []

        with torch.no_grad():

            for i, (x, y) in enumerate(self.test_data_loader):

                shape_ = x.shape[-1]
                # sync with training

                x = x.unsqueeze(1).float().to(self.device)
                y = y.unsqueeze(1).float().to(self.device)

                pred = self.model(x)
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                # np.save(os.path.join(self.save_path, 'x', '{}_result'.format(i)), x)
                # np.save(os.path.join(self.save_path, 'y', '{}_result'.format(i)), y)
                # np.save(os.path.join(self.save_path, 'pred', '{}_result'.format(i)), pred)
                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_psnr_avg1.append(ori_psnr_avg / len(self.test_data_loader))
                ori_ssim_avg += original_result[1]
                ori_ssim_avg1.append(ori_ssim_avg / len(self.test_data_loader))
                ori_rmse_avg += original_result[2]
                ori_rmse_avg1.append(ori_rmse_avg / len(self.test_data_loader))
                pred_psnr_avg += pred_result[0]
                pred_psnr_avg1.append(pred_psnr_avg / len(self.test_data_loader))
                pred_ssim_avg += pred_result[1]
                pred_ssim_avg1.append(pred_ssim_avg / len(self.test_data_loader))
                pred_rmse_avg += pred_result[2]
                pred_rmse_avg1.append(pred_rmse_avg / len(self.test_data_loader))

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

            logger.debug('Original\nPSNR avg: {:.4f} , SSIM avg: {:.4f} , RMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.test_data_loader), ori_ssim_avg / len(self.test_data_loader),
                ori_rmse_avg / len(self.test_data_loader)))
            logger.debug('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.test_data_loader), pred_ssim_avg / len(self.test_data_loader),
                pred_rmse_avg / len(self.test_data_loader)))