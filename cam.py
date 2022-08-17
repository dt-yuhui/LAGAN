import argparse
import cv2
from skimage import morphology
import numpy as np
import torch
import os
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from models.la_module import ClassifyModel
from options import get_opt
from utils import util
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def preprocess_image(img):
    preprocessing = A.Compose(
                [
                    A.Resize(256, 256),
                    A.Normalize(mean=(0.5,), std=(0.5, )),
                    ToTensorV2(),
                ]
    )

    img = preprocessing(image=img.copy())['image']
    return img.unsqueeze(0)


def get_cam(opt):

    if opt.attention_mode == 'soft':

        Model = ClassifyModel(opt)
        Model.load(opt.save_epoch)
        model = Model.model
        model.eval()

        gradcam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=True)
        images = []
        img_dir = os.path.join(opt.dataroot, opt.phase, 'img', 'B')
        cube_list = os.listdir(img_dir)
        for cube in cube_list:
            bscan_list = os.listdir(os.path.join(img_dir, cube))
            for bscan in bscan_list:
                img_path = os.path.join(img_dir, cube, bscan)
                images.append(img_path)

        for img_path in images:
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img.copy(), (256, 256))
            rgb_img = np.float32(cv2.merge([img, img, img])) / 255
            input_tensor = preprocess_image(img).cuda()

            grayscale_cam = gradcam(input_tensor=input_tensor,
                                    target_category=1,
                                    aug_smooth=True,
                                    eigen_smooth=True)
            grayscale_cam = gradcam.scale_cam_image(grayscale_cam)

            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            img_path = img_path.split('/')
            cube_id, bscan_id = img_path[-2], img_path[-1]

            cam_save_path = os.path.join(opt.checkpoints_dir, opt.dataset_name, 'cam', opt.model, opt.phase, cube_id)
            util.mkdir(cam_save_path)
            cv2.imwrite(os.path.join(cam_save_path, bscan_id), cam_image)

            gray_cam_save_path = os.path.join(opt.checkpoints_dir, opt.dataset_name, 'gray_cam', opt.model, opt.phase, cube_id)
            util.mkdir(gray_cam_save_path)
            grayscale_cam = np.uint8(grayscale_cam * 255.0)
            cv2.imwrite(os.path.join(gray_cam_save_path, bscan_id), grayscale_cam)

    elif opt.attention_mode == 'hard':
        '''
        generate binary mask
        '''
        gray_cams = []
        cam_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, 'gray_cam', opt.model, opt.phase)
        cube_list = os.listdir(cam_dir)
        for cube in cube_list:
            bscan_list = os.listdir(os.path.join(cam_dir, cube))
            for bscan in bscan_list:
                img_path = os.path.join(cam_dir, cube, bscan)
                gray_cams.append(img_path)
        for cam_path in gray_cams:
            gray_cam = cv2.imread(cam_path, 0)
            threshold, otsu_result = cv2.threshold(gray_cam, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            final_result = morphology.convex_hull_image(otsu_result) * 255

            img_path = cam_path.split('/')
            cube_id, bscan_id = img_path[-2], img_path[-1]

            save_path = os.path.join(opt.checkpoints_dir, opt.dataset_name, 'pseudo_mask', opt.model, opt.phase, cube_id)
            util.mkdir(save_path)
            cv2.imwrite(os.path.join(save_path, bscan_id), final_result)
    else:
        raise NotImplementedError(f'attention mode {opt.attention_mode} is not implemented')


if __name__ == '__main__':

    opt, parser = get_opt()
    opt.dataroot = 'YOUR_DATASET_ROOT'
    opt.dataset_name = 'YOUR_DATASET_NAME'
    opt.experiment_name = 'res2net'
    opt.model = 'res2net'
    opt.attention_mode = 'hard'
    opt.phase = 'train'

    get_cam(opt)
