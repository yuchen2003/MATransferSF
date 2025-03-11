CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=gymma_collect --offline_data_quality=expert --num_episodes_collected=4000 --stop_return=0.9 --seed=1 with env_args.time_limit=50 env_args.key=lbforaging:Foraging-5x5-2p-3f-coop-v2 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=gymma_collect --offline_data_quality=expert --num_episodes_collected=4000 --stop_return=0.9 --seed=1 with env_args.time_limit=50 env_args.key=lbforaging:Foraging-5x5-3p-3f-coop-v2 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=gymma_collect --offline_data_quality=expert --num_episodes_collected=4000 --stop_return=0.9 --seed=1 with env_args.time_limit=50 env_args.key=lbforaging:Foraging-5x5-4p-2f-coop-v2 &
wait

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

#* w在优化过程趋于0，应该是因为没有限制w的值域，例如||w||=1，导致r_loss优化时w变得任意小，考虑加入该限制
"phi's leaky|quick solve w"
#* leaky可能更好一些，基本差不多；减少w优化步数可能影响值函数稳定性，但是会加快比较多，可能考虑间隔优化w
"leaky + period"
#*leaky影响基本不大，period可以达到加速的效果，并维持每一步都解w的性能；但这些trick都无法根本改进性能，当前（2025-02-16_17-33-07）所表现的有少量过拟合，总体是不良泛化

#* 2.18 phi的数量增加仍然导向与单个phi相近的性能，这可能类似于moe中的专家区分度不大的问题；抛开moe来看，也许能从每个phi head的参数分布角度来看分化度 # TODO ablation
"16x32 phi wo w"
#* 去除w cond将使学习变慢，但4p2f上最终性能更好；使用了w作为轨迹特征的case表现出不稳定或性能大幅下降(skill 8x64; 16x32 with w)
“pretrain”
#* pretrain是为了建立可用的内表征，但轨迹本身64维并非很高维，也许无需这样建模；而pretrain之后再进行offline性能低很多，可能是pretrain所学到的模式并不适配于offline阶段的需求，因此仅考虑直接通过offline训练encoder，而抛弃辅助作用的decoder

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 16x32 wo w"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 16x32"

"dense 16x32 vs. skill 8x64 vs. skill 16x32 vs. skill 64x16", all wo w
#* skill相比dense会显著提升4p2f学习速度；而继续增大phi_dim边际效应明显，且可扩展性受限（由于计算方式，phi多时需要复制的张量大小显著增加，这是一个问题;但是尽管显存使用大，计算速度上并没有影响，应该是得益于张量并行运算） #FIXME 之后可能尝试改进；#TODO 另外进行phi_dim比较详细的ablation和分析

#TODO 也许直接实现并尝试一下在线微调，仅offline测试的还是离线多任务性能（并且注意对zeroshot任务是**随机初始化w下的性能**，并非求解w之后的性能）；仅求w或同时也微调架构
#? 猜测：之前seed1和3407的case都发生在1phi的情况下，可能对应学到了良泛化的任务域值函数，并且其w是通用的，所以总体性能都好；当前(2-18_152336, 2-18_171920)在4p2f上的性能可能正对应这种情况，但其中的w在相似任务下通用
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 8x64 online"

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 8x64 finetune"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 8x64 finetune only w"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 8x64 finetune no agentloss"
#* 完整finetune的效果最好，但仍然表现出一般off2on中的各种现象，如unlearn，速度慢等

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_mt --seed=2 --use_wandb=True --wandb_note="skill 8x64 mto"

#* 尝试从ref当中发现良性泛化的影响因素
CUDA_VISIBLE_DEVICES=1 python src.ref/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="1 off task"

#* ⭐️新网络架构
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="new arch"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="new arch"
#* 纯random feature不行，但使用attn_feature+随机初始化固定矩阵降维是可行的
#* phi, mixer, gt state, cross-agent都是CT阶段的信息，DE阶段则不使用
#* explainer使用linear, softmax, tanh, shifted tanh输出头：包含负值w输出的，即linear, tanh在训练后期（35k左右开始，性能基本收敛）会出现大幅波动，此时Q值发散，推测可能是w的正负相消，导致psi输出大值，走向过拟合并逐渐发散；参数正则化可能解决这一问题，并允许带有负值的w，但目前看调整输出头已足够
#* 非归一化的shifted tanh和归一化的softmax，与linear输出的psi，共同都可以调整Q输出值域，使Q值估计收敛到一定值，但shifted tanh存在过估计，性能却较好
#* softmax倾向于输出均匀的w，实际上可能损失了语义区分，影响泛化；softmax和shift tanh分别低估和高估Q值，但是性能仍然不错？
#* h|tau作为特征可能存在特征坍缩的问题，造成值函数发散和性能消失；原因可能是h自回归的特性，使长程误差累计，影响psi学习；但结合softmax头输出的w则性质良好
#? tanh头的这种效应使得Q值可以有比较大的变化，结合其phi和w的多样性，可能有利于off2on
#* softmax+attn, softmax+h, tanh+attn分布内和分布外任务上性能基本一致

#* 尝试online 
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="online"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="online"
#* 基任务没有unlearn，4p2f上快速对齐Q值，性能正常(500 traj)，但w未起作用；3p3f在1k补收敛，性能等同于off阶段性能，说明也没有learning发生，说明当前on阶段task explain的方式过于弱，可能有利于快适应，但不适应继续学习（w之间差别不大说明task explain不会起到作用，但这应该是特征层的问题）

curl https://p.nju.edu.cn/api/portal/v1/login -X POST -d'{"username":"211240066","password":"caqjew-tufhas-0piFce"}'
// "args": [
//     "--online",
//     "--config=qmix",
//     "--env-config=gymma",
//     "--seed=1",
//     "--t_max=40500",
//     "--use_wandb=False",
// ],
// "args": [
//     "--offline",
//     "--config=bc",
//     "--env-config=sc2_offline",
//     "--seed=1",
//     "--t_max=40500",
//     "--use_wandb=False",
// ],
// "args": [
//     "--mto",
//     "--config=mt_qmix_cql",
//     "--env-config=gymma_offline",
//     "--task-config=lbf_test",
//     "--seed=1",
//     "--use_wandb=False",
// ]