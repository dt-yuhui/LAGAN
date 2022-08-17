import os
import sys
import torch
from tqdm import tqdm

from options import get_opt, print_options
from dataloaders.dataset import CustomDataloader, yield_data
from utils.summaries import TensorboardSummary
from models.la_gan import lagan


def training_loop():
    opt, parser = get_opt()
    opt.phase = 'train'
    opt.isTrain = True
    opt.isDebug = False
    print_options(opt, parser)

    normal_dataloader, normal_sampler = CustomDataloader(opt, 'A')
    abnormal_dataloader, _ = CustomDataloader(opt, 'B')
    step_per_epoch = len(abnormal_dataloader)
    model = lagan(opt)
    normal_data_yield = yield_data(normal_dataloader, normal_sampler)

    summary = TensorboardSummary(opt)   # create a summary that display images and losses
    writer = summary.create_summary()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        model.update_learning_rate()
        with tqdm(abnormal_dataloader, desc=f'train_epoch_{epoch}', file=sys.stdout) as tbar:
            for i, abnormal_data in enumerate(tbar):
                global_step = i + step_per_epoch * (epoch - 1)

                normal_data = next(normal_data_yield)
                data = {'abnormal': abnormal_data, 'normal': normal_data}

                model.set_input(data)
                model.optimize_parameters()

                losses = model.get_current_losses()
                # print training losses on tensorboard
                for loss_name, loss in losses.items():
                    writer.add_scalar(f'train/{loss_name}', loss, global_step)

        visuals = model.get_current_visuals()
        summary.visualize_image(writer, visuals, epoch)
    model.save(opt.save_epoch)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True

    training_loop()
    print('Training was successfully finished')
