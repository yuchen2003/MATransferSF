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

#* 尝试多阶段训练
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain with state"
#* 梯度消失；debug发现state token proj时值趋向于变大，经过动作头之后接近发散，此时梯度消失
#* 翻转obs|state QKV；不用state
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain wo state"
#* 翻转后仍然会 xx梯度爆炸xx 梯度消失，可能是对state的处理不当；仅用obs损失更低，考虑使用该设置
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 32 lr 5e-4" --pretrain_batch_size=32 &&
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 16 lr 5e-4" --pretrain_batch_size=16 &&
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 8 lr 5e-4" --pretrain_batch_size=8
#* 32或64更稳定 -> 64 + 大学习率

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 32 lr 5e-3" --pretrain_batch_size=32 --lr=0.005 &&
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 32 lr 1e-3" --pretrain_batch_size=32 --lr=0.001 &&
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 32 lr 1e-4" --pretrain_batch_size=32 --lr=0.0001
#* 1e-3或5e-3这样较大的学习率会更好，收敛更快，Adam自动调整应该起到主要作用；lr再大不行了

#* 64 + 大学习率
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 64 lr 1e-2" --lr=0.01 &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 64 lr 2e-2" --lr=0.02 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 64 lr 5e-2" --lr=0.05 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 64 lr 1e-1" --lr=0.1 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="invdyn pretrain bs 64 lr 5e-3" --lr=0.005
#* 32 + 5e-3 vs. 64 + 5e-3基本相近，前者能省一点资源

#* 稳定性
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="invdyn pretrain stability" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="invdyn pretrain stability" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="invdyn pretrain stability" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="invdyn pretrain stability" &
wait
#* OK.
#* 0314-143006但是现在更长的训练(18w steps)将出现loss大幅上涨到一个平台，之后(22w)恢复正常loss，原因未知。还有0314-105349

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="invdyn pretrain more param"
#* 但是增加action head的参数就会造成梯度消失
#* 更少参数（单层线性层）收敛到次优
#* 尝试向phi模块增加参数，参考transformer的设计，拓展attention后映射模块，但**去掉所有残差链接**，性能最好，并且稳定。有趣的是，不同的几种phi_gen设计在pretrain阶段展示出相同的loss曲线形状（波动等）
#* 以上对比0312-234144; 0314-105349; 0314-111314
#* phi_gen后训练效果：(1. 0314-140954：Adam后训练, 2. 0314-143006：Adam从头训练, 3. 0314-144620：SGD后训练)
#* Adam出现loss暴涨后恢复，SGD没有出现loss大幅上涨的情况，SGD略好于Adam，但总体都没有提升

#* 不同seed又出现之前那种意外性能的情况
#? pretrain seed和offline seed一致性？
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="invdyn pretrain seed 2" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="invdyn pretrain seed 3" &
wait
#* 良性泛化下的w能保持较低的方差，即比较固定
#* w_explainer不太靠谱，应该有很严重的过拟合：
#! 如果不用r信号学习w，则退化为模仿学习，此时有相当于行为策略的性能

#? 从target_mean和Qmean上看，可能发生了保守估计问题
#* 似乎seed的选取比架构和cqlalpha超参的选取都大

#* QPLEX + w regression新架构
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex pretrain" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="qplex pretrain" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="qplex pretrain" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="qplex pretrain" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex offline" --checkpoint_path="/home/amax/xyc/MATr/offpymarl/results/transfer/lbforaging/lbf_test/lbforaging:Foraging-5x5-2p-2f-coop-v2-expert+lbforaging:Foraging-5x5-3p-2f-coop-v2-expert+lbforaging:Foraging-5x5-2p-3f-coop-v2-expert+lbforaging:Foraging-5x5-4p-2f-coop-v2-expert/tr_sf/seed_1_tr_sf_2025-03-18_16-52-43/models/pretrain" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="qplex offline" --checkpoint_path="/home/amax/xyc/MATr/offpymarl/results/transfer/lbforaging/lbf_test/lbforaging:Foraging-5x5-2p-2f-coop-v2-expert+lbforaging:Foraging-5x5-3p-2f-coop-v2-expert+lbforaging:Foraging-5x5-2p-3f-coop-v2-expert+lbforaging:Foraging-5x5-4p-2f-coop-v2-expert/tr_sf/seed_2_tr_sf_2025-03-18_16-52-43/models/pretrain" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="qplex offline" --checkpoint_path="/home/amax/xyc/MATr/offpymarl/results/transfer/lbforaging/lbf_test/lbforaging:Foraging-5x5-2p-2f-coop-v2-expert+lbforaging:Foraging-5x5-3p-2f-coop-v2-expert+lbforaging:Foraging-5x5-2p-3f-coop-v2-expert+lbforaging:Foraging-5x5-4p-2f-coop-v2-expert/tr_sf/seed_3_tr_sf_2025-03-18_16-52-43/models/pretrain" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="qplex offline" --checkpoint_path="/home/amax/xyc/MATr/offpymarl/results/transfer/lbforaging/lbf_test/lbforaging:Foraging-5x5-2p-2f-coop-v2-expert+lbforaging:Foraging-5x5-3p-2f-coop-v2-expert+lbforaging:Foraging-5x5-2p-3f-coop-v2-expert+lbforaging:Foraging-5x5-4p-2f-coop-v2-expert/tr_sf/seed_4_tr_sf_2025-03-18_16-52-43/models/pretrain" &
wait

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex offline phi origin mixer"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex offline phi origin mixer w cql"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="phi origin mixer, w cql, no reg"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex off simplified"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex off simplified r_hat"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex off simplified r_hat no std"
#* 暂时调下来没有用，无性能，可能mixer里面有一些问题

CUDA_VISIBLE_DEVICES=1 python src.attn/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qattn"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qattn param()"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="qattn param() seed 2"

#* 0319-190917 & 0319-180514 等 可验证linear mixer使得phi-mix与q-mix等价；linear, elu, tanh

#* 0319-213933 seed 2能上!! 尽管不清楚原因，可能是改正的tdlearning
#* linear由于phi和w的值域限制，对值估计不太好；加上nonlinear后值估计又容易高估或是发散（tanh过保守，leaky发散）
#* r_loss没有什么变化，考虑到对任务id的初始化，w在学习过程中并未起到作用；通过调试观察w值，w取为固定的onehot，与w相关统计量对应，说明并未起作用