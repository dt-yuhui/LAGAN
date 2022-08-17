import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
from collections import OrderedDict
from models import res2net
from models.networks import get_scheduler


class ClassifyModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = 'cuda' if not opt.use_cpu else 'cpu'
        self.model_name = opt.model
        self.initial_net()
        self.loss_names = ['ce', 'acc']
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.scheduler = get_scheduler(self.optimizer, opt)

    def set_input(self, input):
        self.input = input
        self.img = input['image'].to(self.device)
        self.gt_label = input['label'].to(self.device)

    def forward(self, input):
        self.set_input(input)
        self.pred_logit = self.model(self.img)

        return self.pred_logit

    def backward(self):
        self.loss_ce = self.criterion(self.pred_logit, self.gt_label)

        self.loss_ce.backward()

    def optimize_parameters(self, input):
        self.forward(input)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def save(self, epoch):
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.experiment_name, f'epoch_{epoch}_net_')
        torch.save(self.model.state_dict(), save_path + self.model_name + '.pth')
        print(f'----save the {self.model_name} at epoch {epoch} successfully')

    def load(self, epoch):
        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name, self.opt.experiment_name, f'epoch_{epoch}_net_')
        state_dict = torch.load(load_path + self.model_name + '.pth', map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f'----load the pretrained {self.model_name} successfully----')

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))

        return errors_ret

    @torch.no_grad()
    def evaluate(self):
        _, self.pred_label = torch.max(self.pred_logit.data, 1)
        self.correct = self.pred_label.eq(self.gt_label.long().data).sum()
        self.loss_acc = 100. * self.correct / np.shape(self.pred_label)[0]

    def initial_net(self):
        self.model = res2net.res2net50_v1b_26w_4s(pretrained=False, num_classes=2).to(self.device)
