python src/main.py --collect --config=qmix --env-config=gymma_collect --offline_data_quality=expert --save_replay_buffer=./replay_buffer --num_episodes_collected=100 --stop_return=0.9 --seed=1 with env_args.time_limit=50 env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v1

python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=False --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=4m --use_wandb=True

python src/main.py --transfer --config=tr_bc --env-config=gymma_transfer --task-config=lbf_test --seed=1 --time_limit=50 --t_max=50200 --online_t_max=10200 --use_wandb=True --wandb_note=test-transfer > /dev/null

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --time_limit=50 --use_wandb=True --wandb_note=test-single-phi

CUDA_VISIBLE_DEVICES=1 python src/main.py --mto --config=tr_bc --env-config=gymma_offline --task-config=lbf_test --seed=1 --time_limit=50 --use_wandb=False --wandb_note=compare-baseline

python src/main.py 
--transfer 
--config={} 
--env-config=gymma_transfer 
--task-config=lbf_test 
--seed=1 
--time_limit=25 
--t_max=50200 
--online_t_max=10200 
--use_wandb=True 
--wandb_note=test-transfer 
