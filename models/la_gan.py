import os
import torch
import torch.nn as nn
from models import networks
from collections import OrderedDict


class lagan(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = 'cuda' if not opt.use_cpu else 'cpu'

        self.loss_names = ['adv_D', 'adv_G']
        self.visual_names = ['real_abnormal', 'fake_normal']
        if self.isTrain:
            self.visual_names.append('real_normal')

            if self.opt.l_mask_l1 > 0:
                self.loss_names.append('mask_l1')

        self.netG = networks.define_G(opt).to(self.device)
        if self.isTrain:
            self.netD = networks.define_D(opt).to(self.device)
            self.create_loss_fns(opt)
            self.create_optimizers(opt)

    def set_input(self, input):
        if self.isTrain:
            self.real_abnormal = input['abnormal']['image'].to(self.device)
            self.real_normal = input['normal']['image'].to(self.device)
            self.mask = input['abnormal']['mask'].to(self.device).unsqueeze(1)
        else:
            self.real_abnormal = input['image'].to(self.device)
            self.image_path = input['image_path']
            self.mask = input.get('mask', torch.zeros((1, self.opt.crop_size, self.opt.crop_size), dtype=torch.uint8)).to(self.device).unsqueeze(1)

    def create_loss_fns(self, opt):
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = nn.L1Loss()

    def create_optimizers(self, opt):
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def forward(self):
        self.fake_normal = self.netG(self.real_abnormal)

    def backward_G(self):
        # generation task loss
        self.loss_adv_G = self.criterionGAN(self.netD(self.fake_normal), True)

        # mask l1 loss
        if self.opt.l_mask_l1 > 0:
            if not self.opt.mask_ablation:
                self.loss_mask_l1 = self.criterionL1((255 - self.mask) / 255 * self.real_abnormal,
                                                     (255 - self.mask) / 255 * self.fake_normal)
            else:
                self.loss_mask_l1 = self.criterionL1(self.real_abnormal, self.fake_normal)
        else:
            self.loss_mask_l1 = 0

        self.loss_G = self.loss_adv_G + self.loss_mask_l1 * self.opt.l_mask_l1
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_adv_D = self.backward_D_basic(self.netD, self.real_normal, self.fake_normal)

    def optimize_parameters(self):
        self.forward()  # compute fake images and reconstruction images.

        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)[0]

        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))

        return errors_ret

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'learning rate = {lr:.7f}')

    def save(self, epoch):
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.experiment_name, f'epoch_{epoch}_net_')
        torch.save(self.netG.state_dict(), save_path + 'G.pth')
        print(f'----save the network at epoch {epoch} successfully')

    def load(self, epoch):
        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.experiment_name, f'epoch_{epoch}_net_')
        state_dict_g = torch.load(load_path + 'G.pth', map_location=self.device)
        self.netG.load_state_dict(state_dict_g)
        print('----load the trained network successfully----')

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
