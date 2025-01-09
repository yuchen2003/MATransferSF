# python src/main.py --collect --config=<alg> --env-config=sc2_collect with env_args.map_name=<map_name> offline_data_quality=<quality> save_replay_buffer=<whether_to_save_replay> num_episodes_collected=<num_episodes_per_collection> stop_winrate=<stop_winrate> --seed=<seed>

import subprocess
from multiprocessing import Pool
import sys

tasks = []
cmds = []
dev_id = 0
time = 0 # Avoid timestamp conflicts

cmd_tpl = "sleep {}; \
CUDA_VISIBLE_DEVICES={} \
python src/main.py \
--collect \
--config={} \
--env-config=sc2_collect \
--offline_data_quality=expert \
--num_episodes_collected=100 \
--save_model_interval=500000 \
--stop_winrate=0.9 \
--seed=1 \
--t_max=40050 \
--map_name={} \
--use_wandb=False \
"

# , (3, 2), (4, 2), (4, 3)
for map in ['3m', '2s3z']:
    tasks.append(map)

for t in tasks:
    for alg in ['maa2c']:
        cmds.append(cmd_tpl.format(time, dev_id % 2, alg, t))
        time += 3
        dev_id += 1

def run_cmd(cmd):
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ret.returncode

if __name__ == '__main__':
    argv = sys.argv
    
    print("Running the following commands parallelly: ")
    for cmd in cmds:
        print('  ', cmd)
    
    
    ret = None
    # if argv[-1] != 'tc': # testcmd
    #     with Pool(processes=len(cmds)) as pool:
    #         ret = pool.map(run_cmd, cmds)
        
    print('Programs done with return codes: ', ret)
    