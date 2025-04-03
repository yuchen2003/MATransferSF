# CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.0001 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-8 lr-0.0001" &
# CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.0002 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-8 lr-0.0002" &
# wait
# CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.0003 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-8 lr-0.0003" &
# CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.00005 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-8 lr-0.00005" &
# wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.0001 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-8 lr-0.0001" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.0002 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-8 lr-0.0002" &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.0003 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-8 lr-0.0003" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.00005 --pretrain_batch_size=8 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-8 lr-0.00005" &
wait

# CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.0001 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-16 lr-0.0001" &
# CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.0002 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-16 lr-0.0002" &
# wait
# CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.0003 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-16 lr-0.0003" &
# CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --lr=0.00005 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2 bs-16 lr-0.00005" &
# wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.0001 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-16 lr-0.0001" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.0002 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-16 lr-0.0002" &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.0003 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-16 lr-0.0003" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=msz_hard --lr=0.00005 --pretrain_batch_size=16 --seed=1234 --use_wandb=True --wandb_note="sc2hard bs-16 lr-0.00005" &
wait
