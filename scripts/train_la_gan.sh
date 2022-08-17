set -ex
python train_la_gan.py \
--dataroot YOUR_DATASET_ROOT --dataset_name YOUR_DATASET_NAME \
--experiment_name YOUR_EXPERIMENT_NAME --model res2net --attention_mode hard --use_sn \
--netG unet_256 --netD MS --gan_mode lsgan --num_D 2 \
--load_size 286 --crop_size 256 --batch_size 8 --phase train \
--n_epochs 100 --n_epochs_decay 100