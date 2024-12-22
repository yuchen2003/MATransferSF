python src/main.py --collect --config=qmix --env-config=gymma_collect --offline_data_quality=expert --save_replay_buffer=./replay_buffer --num_episodes_collected=100 --stop_return=0.9 --seed=1 with env_args.time_limit=50 env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v1

python src/main.py --transfer --config=tr_bc --env-config=gymma_transfer --task-config=lbf_test --seed=1 --time_limit=50 --t_max=50200 --online_t_max=10200 --use_wandb=True --wandb_note=test-transfer > /dev/null

python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --time_limit=50 --t_max=50200 --online_t_max=10200 --use_wandb=True --wandb_note=test-transfer-pretrain

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
