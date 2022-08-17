set -ex
python test_la_module.py \
--dataroot YOUR_DATASET_ROOT --dataset_name YOUR_DATASET_NAME \
--experiment_name YOUR_EXPERIMENT_NAME --model res2net \
--load_size 256  --batch_size 1 --num_threads 0 \
--phase test