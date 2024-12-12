# python src/main.py --collect --config=qmix --env-config=gymma_collect --offline_data_quality=expert --save_replay_buffer=./replay_buffer --num_episodes_collected=100 --stop_return=0.9 --seed=1 with env_args.time_limit=50 env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v1

import subprocess
from multiprocessing import Pool
import gym
import lbforaging
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
--env-config=gymma_collect \
--offline_data_quality=expert \
--num_episodes_collected=5000 \
--save_model_interval=500000 \
--stop_return=0.9 \
--seed=1 \
--time_limit=25 \
--t_max=4005000 \
--key=lbforaging:{} \
"
# , (3, 2), (4, 2), (4, 3)
for p, f in [(2, 1), (2, 2), (3, 2)]:
    tasks.append(f"Foraging-5x5-{p}p-{f}f-coop-v2")

for t in tasks:
    for alg in ['qmix', 'maa2c']:
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
    if argv[-1] != 'testcmd':
        with Pool(processes=len(cmds)) as pool:
            ret = pool.map(run_cmd, cmds)
        
    print('Programs done with return codes: ', ret)
    
