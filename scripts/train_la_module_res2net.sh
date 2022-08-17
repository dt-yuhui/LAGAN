set -ex
python train_la_module.py \
--dataroot YOUR_DATASET_ROOT --dataset_name YOUR_DATASET_NAME \
--experiment_name YOUR_EXPERIMENT_NAME --model res2net \
--load_size 286 --crop_size 256 --batch_size 50 --phase train \
--n_epochs 100 --n_epochs_decay 100