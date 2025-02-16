python src/main.py --collect --config=qmix --env-config=gymma_collect --offline_data_quality=expert --save_replay_buffer=./replay_buffer --num_episodes_collected=100 --stop_return=0.9 --seed=1 with env_args.time_limit=50 env_args.key=lbforaging:Foraging-8x8-2p-2f-coop-v1

python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=False --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=4m --use_wandb=True

python src/main.py --transfer --config=tr_bc --env-config=gymma_transfer --task-config=lbf_test --seed=1 --time_limit=50 --t_max=50200 --online_t_max=10200 --use_wandb=True --wandb_note=test-transfer > /dev/null

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --time_limit=50 --use_wandb=True --wandb_note=test-offline

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note=test-full

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note=test-pretrain

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_bc --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note=compare-baseline

CUDA_VISIBLE_DEVICES=1 python src/main.py --mto --config=tr_bc --env-config=gymma_offline --task-config=lbf_test --seed=1 --time_limit=50 --use_wandb=False --wandb_note=compare-baseline

CUDA_VISIBLE_DEVICES=1 python src/main.py --mto --config=mt_bc --env-config=sc2_offline --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="test mt_bc for transfer"
CUDA_VISIBLE_DEVICES=1 python src/main.py --mto --config=mt_qmix_cql --env-config=gymma_offline --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="test mt_qmix_cql for transfer"
CUDA_VISIBLE_DEVICES=1 python src/main.py --mto --config=mt_vdn --env-config=gymma_offline --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="test mt_vdn for transfer"
#* 再一次说明了LVF方法的不稳定性

CUDA_VISIBLE_DEVICES=1 python src/main.py --mto --config=mt_qmix_cql --env-config=gymma_offline --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="test mt_qmix without cql for transfer"
#* CQL对于离线方法很重要，但QPLEX可能可以解决off所带来的ood和偏移问题？

# test performance
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note=test-full 

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note=test-full-1-phi
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note=test-full-1-phi
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note=small-pretrain-batch-64


CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note=test-no-pretrain
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note=test-no-pretrain

#* (online GPI) too slow, should contain very few online steps; without pretrain & offline, phi_loss or r_loss can hardly be learned; Besides, VDN is proven to behave divergently under offline paradigm, qplex mixer should be used
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note=test-only-online
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note=test-only-online

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="test psi learning"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="test psi learning"

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="test psi learning with CQL"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="test psi learning off with CQL"

# test tr_sf multitask transfer online w/ CQL； test tr_sf multitask transfer online 
--note="tr_sf online single task with mto fix-eps"
#* 也许简单的任务如lbf无需online learn。transfer+pretrain+gymma_lbforaging:Foraging-8x8-2p-2f-coop-v2+seed_1_tr_sf_2025-01-07_11-25-04；可以纯靠零样本泛化。

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="tr_sf online single task"
#* 250106的online run中epsilon设定过大，所以存在问题
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="tr_sf online single task fix-eps"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="tr_sf online single task with mto fix-eps"
#* offmt多训练会on的时候会更稳定；5w步offmt足够
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="tr_sf online single task with eps==0"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="tr_sf online multi task with eps==0"
#* off2on 中在线多任务表现尚可，lbf上存在少量改进空间，之前的问题均由epsilon设定造成（应理解为经过离线预训练的模型无需探索，而能有一定的初始性能）；另一个问题是，lbf上的jump start性能都更好，有可能无需online阶段
#* 从实际运行时间来看，online mt速度快很多

# test on sc2
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="tr_sf sc2 mton"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="tr_sf sc2 ston"

# without phi, new perspective, universal value decomposition(SF是universal value，用在MA中则是UVDecomp，广义值分解)
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="new tr_sf" 
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="new tr_sf"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="new tr_sf 1 phi"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="new tr_sf 1 phi stoc z_w"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="new tr_sf 4 phi"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="new tr_sf skill 4 phi"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="skill 4 phi w"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="skill 8 phi w small model"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="skill 8 phi w small model"
#* 模型可以学起来，但是性能相比Qlearning差很多，原则上多个psi不会比单个Q更差；需进一步改进模型细节
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="skill pretrain"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="skill offline"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="skill offline no pretrain"
#* 分离训练过程不可行，也许要更精细的frozen控制
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="skill 64 phi"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="skill 1 phi"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="old 1 phi"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="debug phi"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="debug phi"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="debug phi seed 1"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="debug phi seed 2"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="debug phi seed 3"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=3407 --use_wandb=True --wandb_note="debug phi seed 3407"

#* 2.14 比对cur和ref没有发现显著差别，但两者性能仍然差很多，原因未知，故打算从ref开始调；相关对比在"debug phi"标记下
#* 训练任务和2f任务上性能都可以提升到最优，而3f任务上差异很大，说明某个因素是影响任务泛化性能（或是对训练任务的过拟合）的关键
#* mixer无影响；task weight的某个流程有问题，可能是多个类属性的共享有问题（更正）；这种泛化性能应该是基础架构提供的
#* 调试发现是agent中的一些冗余模块造成的，这些冗余模块可能对参数初始化和优化过程造成了影响，从而影响了学习过程，本质上是核心模型鲁棒性较差
#* 仅调整seed也产生了相近现象，说明确实是不鲁棒的问题，这一点可以作为基座性能的一点观察，如果能改进则是本项目的一个贡献

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="phi's"

# curl https://p.nju.edu.cn/api/portal/v1/login -X POST -d'{"username":"211240066","password":"caqjew-tufhas-0piFce"}'
