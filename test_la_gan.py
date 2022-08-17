import os
import torch
import numpy as np
from utils import util
from models.la_gan import lagan
from options import get_opt
from dataloaders.dataset import CustomDataloader
from tqdm import tqdm
import cv2


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    opt, parser = get_opt()

    opt.results_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.experiment_name, 'results')
    opt.isTrain = False

    util.mkdir(opt.results_dir)
    model = lagan(opt)
    model.load(opt.save_epoch)
    model.eval()

    abnormal_dataloader, _ = CustomDataloader(opt, 'B')
    abnormal_tbar = tqdm(abnormal_dataloader)

    with torch.no_grad():
        for i, data in enumerate(abnormal_tbar):
            model.set_input(data)  # unpack data from data loader
            model.forward()          # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = data['image_path'][0].split('/')
            gt = np.transpose(data['mask'].data.cpu().numpy(), (1, 2, 0))
            cube_id, bscan_id = img_path[-2], img_path[-1]

            real_img, fake_img = visuals['real_abnormal'], visuals['fake_normal']
            _real_img, _fake_img = map(util.tensor2im, (real_img, fake_img))

            test_img = np.concatenate((_real_img, _fake_img), axis=1)

            residual_img = np.expand_dims(cv2.absdiff(_real_img, _fake_img), axis=0)  # 1x256x256
            _residual_img = np.transpose(residual_img, (1, 2, 0))    # 256x256x1

            _test_img = np.concatenate((test_img, gt, _residual_img), axis=1)

            test_save_path = os.path.join(opt.results_dir, 'test', cube_id)
            residual_save_path = os.path.join(opt.results_dir, 'residual', cube_id)
            generation_save_path = os.path.join(opt.results_dir, 'generation', cube_id)
            util.mkdir(test_save_path)
            util.mkdir(residual_save_path)
            util.mkdir(generation_save_path)
            cv2.imwrite(os.path.join(test_save_path, bscan_id), _test_img)
            cv2.imwrite(os.path.join(residual_save_path, bscan_id), _residual_img)
            cv2.imwrite(os.path.join(generation_save_path, bscan_id), _fake_img)
