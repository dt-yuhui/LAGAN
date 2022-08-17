import os
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(opt, mode):
    img_dir = os.path.join(opt.dataroot, opt.phase, 'img', mode)
    cube_list = os.listdir(img_dir)
    if mode == 'B':
        images, gt = [], []
        assert os.path.isdir(opt.dataroot), f'{opt.dataroot} is not a valid directory'

        if opt.isTrain:
            # during training process, we only use the pseudo edema area
            # ablation study
            gt_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, 'pseudo_mask', opt.model, opt.phase)
        else:
            # ground truth for testing
            gt_dir = os.path.join(opt.dataroot, opt.phase, 'gt')
        for cube in cube_list:
            bscan_list = os.listdir(os.path.join(gt_dir, cube))
            for bscan in bscan_list:
                img_path, gt_path = os.path.join(img_dir, cube, bscan), os.path.join(gt_dir, cube, bscan)
                images.append(img_path)
                gt.append(gt_path)
        print(f'Find {len(images)} images in {img_dir}')
        assert (len(images) == len(gt)), 'length of images and ground truth should be the same'
        return images, gt

    # there're no masks for normal ones
    else:
        images = []
        assert os.path.isdir(opt.dataroot), f'{opt.dataroot} is not a valid directory'

        for cube in cube_list:
            bscan_list = os.listdir(os.path.join(img_dir, cube))
            for bscan in bscan_list:
                img_path = os.path.join(img_dir, cube, bscan)
                images.append(img_path)
        print(f'Find {len(images)} images in {img_dir}')
        return images


def get_transform(opt):
    if opt.isTrain:
        trans = A.Compose(
            [
                A.Resize(opt.load_size, opt.load_size),
                A.RandomCrop(opt.crop_size, opt.crop_size),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.5,), std=(0.5, )),
                ToTensorV2(),
            ]
        )
    else:
        trans = A.Compose(
            [
                A.Resize(opt.load_size, opt.load_size),
                A.Normalize(mean=(0.5,), std=(0.5, )),
                ToTensorV2(),
            ]
        )
    return trans
