import numpy as np
import cv2
import os
import skimage.morphology as sm
from utils import util
from options import get_opt
from pytorch_grad_cam.utils.image import show_cam_on_image


def cal_criterion(pred, gt):
    smooth = 1.
    m1 = pred.reshape(-1)
    m2 = gt.reshape(-1)
    m1 = np.where(m1 > 0, 1, 0)
    m2 = np.where(m2 > 0, 1, 0)
    intersection = (m1 * m2).sum()
    union = m1.sum() + m2.sum() - intersection
    fn = ((1-m1) * m2).sum()
    fp = ((1-m2) * m1).sum()

    dice = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    iou = intersection / union
    fnr = fn / union
    fpr = fp / union

    return dice, iou, fnr, fpr


def cal_background_preserved(otsu, gt):
    m1 = otsu.reshape(-1)
    m2 = gt.reshape(-1)
    m1 = np.where(m1 > 0, 1, 0)
    m2 = np.where(m2 > 0, 1, 0)
    fp = ((1-m2) * m1).sum()

    return fp


class PostProcessor:
    def __init__(self, opt):
        super().__init__()
        self.result_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.experiment_name, 'results')
        self.mode = opt.phase
        self.residual_dir = os.path.join(self.result_dir, 'residual')
        self.otsu_dir = os.path.join(self.result_dir, 'otsu')
        self.seg_dir = os.path.join(self.result_dir, 'seg')
        self.contour_dir = os.path.join(self.result_dir, 'contour')
        self.saliency_dir = os.path.join(self.result_dir, 'saliency')

        util.mkdir(self.otsu_dir)
        util.mkdir(self.seg_dir)
        util.mkdir(self.contour_dir)
        util.mkdir(self.saliency_dir)

        self.bscan_dir = os.path.join(opt.dataroot, opt.phase, 'img/B')
        self.cube_list = os.listdir(self.bscan_dir)
        self.gt_dir = os.path.join(opt.dataroot, opt.phase, 'gt')

    def get_otsu(self):
        for cube in self.cube_list:
            bscan_list = os.listdir(os.path.join(self.residual_dir, cube))
            util.mkdir(os.path.join(self.otsu_dir, cube))
            for bscan in bscan_list:
                res_path = os.path.join(self.residual_dir, cube, bscan)
                res_img = np.array(cv2.imread(res_path, 0))

                ret, binary = cv2.threshold(res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(self.otsu_dir, cube, bscan), binary)

    def get_segmentation(self):
        for cube in self.cube_list:
            bscan_list = os.listdir(os.path.join(self.otsu_dir, cube))
            for bscan in bscan_list:
                otsu_path = os.path.join(self.otsu_dir, cube, bscan)
                otsu_img = np.array(cv2.imread(otsu_path, 0))

                if not os.path.exists(os.path.join(self.seg_dir, cube)):
                    util.mkdir(os.path.join(self.seg_dir, cube))
                dst = sm.remove_small_objects(otsu_img == 255, min_size=64, connectivity=2) * 1
                seg_result = sm.convex_hull_image(dst) * 255
                cv2.imwrite(os.path.join(self.seg_dir, cube, bscan), seg_result)

    def eval_segmentation(self):
        dice_array, iou_array, fnr_array, fpr_array = [], [], [], []
        with open(os.path.join(self.result_dir, 'seg_result.csv'), 'w+') as f:
            f.seek(0)
            f.truncate()
            f.write('cube,bscan,dice,or,fnr,fpr\n')
        for cube in self.cube_list:
            bscan_list = os.listdir(os.path.join(self.seg_dir, cube))
            for bscan in bscan_list:
                seg_path = os.path.join(self.seg_dir, cube, bscan)
                gt_path = os.path.join(self.gt_dir, cube, bscan)
                pred_seg = np.array(cv2.imread(seg_path, 0))
                gt = np.array(cv2.resize(cv2.imread(gt_path, 0), (256, 256)))

                dice, iou, fnr, fpr = cal_criterion(pred_seg, gt)
                dice_array.append(dice)
                iou_array.append(iou)
                fnr_array.append(fnr)
                fpr_array.append(fpr)

                with open(os.path.join(self.result_dir, 'seg_result.csv'), 'a') as f:
                    f.write(f'{cube},{bscan},{dice},{iou},{fnr},{fpr}\n')

        dice_array, iou_array, fnr_array, fpr_array = np.array(dice_array), np.array(iou_array), np.array(fnr_array), np.array(fpr_array)
        with open(os.path.join(self.result_dir, 'result.csv'), 'w+') as f:
            f.write(f'dice: {dice_array.mean() * 100} +- {dice_array.std()*100}\n'
              f'or: {iou_array.mean() * 100} +- {iou_array.std()*100}\n'
              f'fnr: {fnr_array.mean() * 100} +- {fnr_array.std()*100}\n'
              f'fpr: {fpr_array.mean()*100} +- {fpr_array.std()*100}')
        print(f'dice: {dice_array.mean() * 100} +- {dice_array.std()*100}\n'
              f'or: {iou_array.mean() * 100} +- {iou_array.std()*100}\n'
              f'fnr: {fnr_array.mean() * 100} +- {fnr_array.std()*100}\n'
              f'fpr: {fpr_array.mean()*100} +- {fpr_array.std()*100}')

    def eval_generation(self):
        fp_array = []
        with open(os.path.join(self.result_dir, 'generation_bscan_result.csv'), 'w+') as f:
            f.seek(0)
            f.truncate()
            f.write('cube,bscan,fp\n')
        for cube in self.cube_list:
            bscan_list = os.listdir(os.path.join(self.seg_dir, cube))
            for bscan in bscan_list:
                otsu_path = os.path.join(self.otsu_dir, cube, bscan)
                gt_path = os.path.join(self.gt_dir, cube, bscan)
                otsu_img = np.array(cv2.imread(otsu_path, 0))
                gt = np.array(cv2.resize(cv2.imread(gt_path, 0), (256, 256)))

                fp = cal_background_preserved(otsu_img, gt)
                fp_array.append(fp)
                with open(os.path.join(self.result_dir, 'generation_bscan_result.csv'), 'a') as f:
                    f.write(f'{cube},{bscan},{fp}\n')

        fp_array = np.array(fp_array)
        with open(os.path.join(self.result_dir, 'generation_result.csv'), 'w+') as f:
            f.write(f'fp: {fp_array.mean()} +- {fp_array.std()}\n')
        print(f'fp: {fp_array.mean()} +- {fp_array.std()}\n')

    def draw_contour(self):
        for cube in self.cube_list:
            bscan_list = os.listdir(os.path.join(self.bscan_dir, cube))
            util.mkdir(os.path.join(self.contour_dir, cube))
            for bscan in bscan_list:
                bscan_path = os.path.join(self.saliency_dir, cube, bscan)
                bscan_img = np.array(cv2.resize(cv2.imread(bscan_path, 1), (256, 256)))
                # gt_path = os.path.join(self.gt_dir, cube, bscan)
                # gt = np.array(cv2.resize(cv2.imread(gt_path, 0), (256, 256)))

                bscan_w_contour = bscan_img.copy()
                pred_seg = cv2.imread(os.path.join(self.seg_dir, cube, bscan), 0)
                contour_pred, hierarchy = cv2.findContours(pred_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                bscan_w_contour = cv2.drawContours(bscan_w_contour, contour_pred, -1, (0, 0, 255), 2)

                # contour_gt, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # bscan_w_contour = cv2.drawContours(bscan_w_contour, contour_gt, -1, (0, 255, 0), 2)

                cv2.imwrite(os.path.join(self.contour_dir, cube, bscan), bscan_w_contour)

    def draw_saliency_map(self):
        for cube in self.cube_list:
            bscan_list = os.listdir(os.path.join(self.residual_dir, cube))
            util.mkdir(os.path.join(self.saliency_dir, cube))
            for bscan in bscan_list:
                res_path = os.path.join(self.residual_dir, cube, bscan)
                res_img = np.float32(cv2.imread(res_path, 0)) / 255
                res_img = util.scale_image(res_img)
                bscan_path = os.path.join(self.bscan_dir, cube, bscan)
                bscan_img = cv2.imread(bscan_path, 0)
                img = cv2.resize(bscan_img.copy(), (256, 256))
                rgb_img = np.float32(cv2.merge([img, img, img])) / 255

                cam_image = show_cam_on_image(rgb_img, res_img, use_rgb=True)
                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.saliency_dir, cube, bscan), cam_image)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt, parser = get_opt()

    opt.dataroot = 'YOUR_DATASET_ROOT'
    opt.dataset_name = 'YOUR_DATASET_NAME'
    opt.experiment_name = f'YOUR_EXPERIMENT_NAME'
    opt.phase = 'test'
    opt.isTrain = False

    postprocessor = PostProcessor(opt)
    # postprocessor.get_otsu()
    # postprocessor.get_segmentation()
    # postprocessor.eval_segmentation()
    # postprocessor.draw_contour()
    # postprocessor.eval_generation()
    # postprocessor.draw_saliency_map()
