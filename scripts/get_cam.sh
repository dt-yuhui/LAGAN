set -ex
python cam.py \
--dataroot YOUR_DATASET_ROOT --dataset_name YOUR_DATASET_NAME \
--model res2net --attention_mode hard --phase train