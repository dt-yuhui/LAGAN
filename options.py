import os
import argparse
from utils import util


def get_opt():
    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--dataroot', type=str, default='YOUR_DATASET_ROOT', help='root to the image dataset')
    parser.add_argument('--experiment_name', type=str, default='YOUR_EXPERIMENT_NAME', help='name of experiment')
    parser.add_argument('--dataset_name', type=str, default='YOUR_DATASET_NAME', help='name of dataset')
    parser.add_argument('--checkpoints_dir', type=str, default='./runs', help='models are saved here')
    parser.add_argument('--use_cpu', action='store_true', help='use this flag to operate in cpu mode')
    # model parameters
    parser.add_argument('--input_nc', type=int, default=1,
                        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=1,
                        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='MS',
                        help='specify discriminator architecture [n_layers | MS]. '
                             'The n_layer model is a 70x70 PatchGAN, MS model is in multi-scale setting.')
    parser.add_argument('--netG', type=str, default='unet_256',
                        help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    parser.add_argument('--num_D', type=int, default=2, help='nums of sub-discriminator in MS discriminator')
    parser.add_argument('--gan_mode', type=str, choices=['lsgan', 'vanilla'], help='type of GAN loss')
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--use_sn', action='store_true',
                        help='whether to use spectral norm in discriminator to stabilize the training process')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--model', type=str, default='res2net',
                        help='# backbone of lesion aware module')
    parser.add_argument('--attention_mode', type=str, default='hard',
                        help='# form of lesion mask, [soft | mask]')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    # dataset parameters
    parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size used for training')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    # training parameters
    parser.add_argument('--save_epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--l_mask_l1', type=float, default=10.0, help='weight of content distortion loss')
    parser.add_argument('--mask_ablation', action='store_true', help='whether to use the lesion prior')

    opt = parser.parse_args()

    return opt, parser


def print_options(opt, parser):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.experiment_name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))

    try:
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    except PermissionError as error:
        print(f'permission error {error}')
        pass
