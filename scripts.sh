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

#* 2.18 phi的数量增加仍然导向与单个phi相近的性能，这可能类似于moe中的专家区分度不大的问题；抛开moe来看，也许能从每个phi head的参数分布角度来看分化度
"16x32 phi wo w"
#* 去除w cond将使学习变慢，但4p2f上最终性能更好；使用了w作为轨迹特征的case表现出不稳定或性能大幅下降(skill 8x64; 16x32 with w)
“pretrain”
#* pretrain是为了建立可用的内表征，但轨迹本身64维并非很高维，也许无需这样建模；而pretrain之后再进行offline性能低很多，可能是pretrain所学到的模式并不适配于offline阶段的需求，因此仅考虑直接通过offline训练encoder，而抛弃辅助作用的decoder

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 16x32 wo w"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="skill 16x32"

"dense 16x32 vs. skill 8x64 vs. skill 16x32 vs. skill 64x16", all wo w
#* skill相比dense会显著提升4p2f学习速度；而继续增大phi_dim边际效应明显，且可扩展性受限（由于计算方式，phi多时需要复制的张量大小显著增加，这是一个问题;但是尽管显存使用大，计算速度上并没有影响，应该是得益于张量并行运算） 
# 之后可能尝试改进（已改进为更好的架构3.19）；
# NOTE 另外进行phi_dim比较详细的ablation和分析（3.21一个隐藏维的维度，取决于phigen所提取特征的内容多少）

# 也许直接实现并尝试一下在线微调，仅offline测试的还是离线多任务性能（并且注意对zeroshot任务是**随机初始化w下的性能**，并非求解w之后的性能）；仅求w或同时也微调架构
# NOTE(3.21一个通用可泛化的表征才是关键，抛开其进行学习并不是可迁移的学习)
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
#* 但sgd对完整训练和后训练都没什么帮助
#* Adam下lr设的很小不会有太大影响，0.001足够
#* 小batch小lr下学习稳定，推测可能是出现了泛化困难和灾难性遗忘问题：模型初期能学习到可泛化表征，过训练使模型过拟合到部分困难样本并大幅遗忘之前的表征（学习率较大但loss曲面平坦，尽管如此，仍存在许多局部最优解）
#* lr <= 0.001, bs <= 32

#* 不同seed又出现之前那种意外性能的情况
#* pretrain seed和offline seed一致性？
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="invdyn pretrain seed 2" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="invdyn pretrain seed 3" &
wait
#* 良性泛化下的w能保持较低的方差，即比较固定
#* w_explainer不太靠谱，应该有很严重的过拟合：
#! 如果不用r信号学习w，则退化为模仿学习，此时有相当于行为策略的性能

#* 从target_mean和Qmean上看，可能发生了保守估计问题（值未对齐+架构和任务与cqclalpha可能需要精确微调）
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

#* 0319-213933 seed 2能上!! 尽管不清楚原因，可能是改正的tdlearning，或是 **parameters()相关的实现不正确**, 目前已采用学习过程中进行detach的做法代替
#* linear由于phi和w的值域限制，对值估计不太好；加上nonlinear后值估计又容易高估或是发散（tanh过保守，leaky发散）
#? 调一下cql_alpha也许可以控制值发散 # CHECK
#* r_loss没有什么变化，考虑到对任务id的初始化，w在学习过程中并未起到作用；通过调试观察w值，w取为固定的onehot，与w相关统计量对应，说明并未起作用
#* unbound weight输出可以减小rloss，但仍然无法进行值对齐；非静态的weight并不是关键因素
#* 乘性特征很容易发散；考虑类似于MAIL的做法进行特征phi学习 
CUDA_VISIBLE_DEVICES=1 python src.qplex/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="qplex"
#* 所有的no cql都学不起来
#* qplex有多任务和低零泛化性能，有待检验online

#* off2on
#* gradnorm初期很大，后续迅速降低，应该是由于值对齐
#* 带eps sched会好一点，有利于探索并增强记忆（简单实验上不显著）

# TODO />
#* ✅pre阶段学phi(phi可以 **纯CT而不用管DE** ), //taskhead (~MATTAR)//，考虑r作为弱监督或predhead目标(某种guidance)，而非具体目标（使用attnhead(over agents)+meanpool，或最大互信息等，避免具体的agent sum），但仍然有可能用linear mixer上phi-mix; 
#* off阶段仅针对phi学习psi值网络和特征提取（注意和phi架构有所不同，额外输入动作）（此时则可能不用考虑拟合reward，mix在psi层面进行），taskhead则固定，或仅加入某种修正项；  
#! 总的来说，phi是作为reward的一个稠密的代理而被使用
#* on阶段仅学习psi值部分，但使用真实reward？；
#* adapt尝试根据MATTAR，设计taskid微调或是其他
# TODO </
#* 不用agent CA也行，SID假设仍然可以用于构建框架

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="new attn pre weighted BCE"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="pre wBCE stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="pre wBCE stable" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="pre wBCE stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="pre wBCE stable" &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="pre wo wBCE stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="pre wo wBCE stable" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="pre wo wBCE stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="pre wo wBCE stable" &
wait
#* //weighted invdyn loss可以缓解之前loss大幅上涨再下降的问题(loss回正6w~20w step不等；改进后则为2w~6w step，上升幅度小得多，较大的上升更晚到来，这可能是优化任务特性)//(只是稍微推迟，且 r guide的第一个实现有问题，不用作对比)，可能可以提升稳定性和泛化性
#* 但10w step后出现暴增的现象仍然存在；10w step的效果比之前的更好，按照之前的做法，仅使用10w step的结果
#* LN > BN，原因未知
#* weighted loss +  lr <= 0.001, bs <= 32效果最好，更长的训练时间内也没有出现暴涨，且具鲁棒性，对seed不敏感；bs=32时收敛最快
#* 没有wBCE也可以，应该主要是学习率的问题，宜小不宜大

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="off stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="off stable" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="off stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="off stable" &
wait
#* mixer activ与性能没有必然关系，偶尔比较好

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --load_step=100023 --use_wandb=True --wandb_note="loadstep 10w"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --load_step=200043 --use_wandb=True --wandb_note="loadstep 20w"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --load_step=300063 --use_wandb=True --wandb_note="loadstep 30w"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --load_step=500103 --use_wandb=True --wandb_note="loadstep 50w"
200043, 300063, 500103
#* 影响不大

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="on stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="on stable" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="on stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="on stable" &
wait
#* 确认off2on可行，大约10ksteps在off次优的任务上到最高

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="off hilp phi" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="off hilp phi" &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="off hilp phi wo wBCE" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=3 --use_wandb=True --wandb_note="off hilp phi wo wBCE" &
wait
#* // learner中的load optim statedict貌似比较重要，原理不清楚 //
#* 主要还是phigen学习的问题

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="pre hilp phi stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="pre hilp phi stable" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="off hilp phi stable" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="off hilp phi stable" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="off hilp phi stable"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="off hilp phi stable"
#* 出现暴涨的pretrain ckpt都学起来了，没出现的都不太学得了，后续可能会展开分析下，作为对phigen的一个研究

#! /> 关于泛化迁移的一个新的见解
#* 在新场景或对新数据进行学习不遗忘的一个关键可能在于特征提取的通用性，即零泛化能力（通用表征等），后续不进行学习仍然可以进行使用，只是针对真实奖励进行学习以对齐价值；这一假设需要检验 # CHECK
#* 作为对比，此前直接将value部分的特征提取层作为phi进行学习有一定的多任务和简单任务的零泛化能力，难一些的任务则无法泛化
#* TDlearning中使用真实rewards和reward_hat(phi)的区别：后者实际上是一个稠密信号，相比稀疏的rewards（例如LBF），更容易学习；但对于混合数据，reward_hat的大小则应当与r对齐，这是w的作用之一，但本质上还应当是phi与r的对齐，因其属于同一函数空间。专家数据下的phi都对应高奖励，此时无对齐也没问题 
#! 就这一点来说，该框架可用于IL|IRL与RL，如果具有专家数据则无需真实奖励，否则需要真实奖励对phi进行奖励对齐
#! 如果这一点可以确证，则能表明SF用于off2on以及更广泛的额可迁移学习的一个全新的解法，并且是十分高效的；这其中的关键则是通用特征phi的学习和构建，以及基于此的psi学习架构和多阶段学习方式；这甚至有可能做到持续学习场景中
#! </ 关于泛化迁移的一个新的见解
#* 所以其实所有的phi^i, psi^i都无需直接对齐奖励值，而是mix之后的phi_tot, psi_tot需要对齐，相对于个体值他们经过mixer的重新组合，而linear mixer保证个体和全局特征语义一致，即前者也满足重新组合的关系；单论前者则是构成了一种pseudo reward（或reward model，等同于IRL中的设定），或者在这里其实是pseudo phi。 ("MAAIRL: recover a reward function that rationalizes the expert behaviors with the least commitment")
#? 对更复杂的任务，也许off阶段需要reward信号帮助调整mixer以实现对齐（简单任务如lbf上没有rloss会稍好一点） # CHECK

#* off带上rloss会有某种错误对齐，导致值估计增加较快，并干扰tdloss的优化，同时gradnorm较大，但有时能进入一些罕见的更优解（2p3f上完成速度更快，有可能是该环境本身存在某种’作弊解‘？）

#smac数据
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=5m_vs_6m --use_wandb=True &
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=9m_vs_10m --use_wandb=True &
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=2s3z --use_wandb=True &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=2s1z_vs_3z --use_wandb=True &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=MMM --use_wandb=True &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_collect --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=MMM2 --use_wandb=True &
wait

#* 适配smacv2
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix_beta --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_6_vs_6 --use_wandb=True
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix_beta --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_6_vs_6 --use_wandb=True
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix_beta --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_6_vs_6 --use_wandb=True
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix_beta --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_6_vs_6 --use_wandb=True
#* parallel的实现可能有问题，或者是并行的方式/64线不适合，或学习率不适合，造成收敛很慢
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix_beta --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_6_vs_6 --use_wandb=True
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix_ft --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_6_vs_6 --use_wandb=True --wandb_note="qmix fted"
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix_ft --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_6_vs_6 --use_wandb=True --wandb_note="qmix fted"
# TODO 目前看smacv2应该是正常的，decomposer之后再看看；目前先收集smacv2数据，调smac性能，以及在lbf上看看whead怎么改性能不掉又合理
# TODO smacv2+qmix大致4m步已收敛，可以提前终止然后用最后的策略收集数据（**仅收集**）
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_5_vs_5 --use_wandb=True &
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_5_vs_6 --use_wandb=True &
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=sc2_v2_protoss --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_protoss_10_vs_10 --use_wandb=True &
wait
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_v2_terran --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_terran_5_vs_5 --use_wandb=True &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_v2_terran --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_terran_5_vs_6 --use_wandb=True &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_v2_terran --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_terran_10_vs_10 --use_wandb=True &
wait
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_v2_zerg --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_zerg_5_vs_5 --use_wandb=True &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_v2_zerg --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_zerg_5_vs_6 --use_wandb=True &
CUDA_VISIBLE_DEVICES=0 python src/main.py --collect --config=qmix --env-config=sc2_v2_zerg --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_zerg_10_vs_10 --use_wandb=True &
wait

CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_v2_zerg --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_zerg_5_vs_6 --use_wandb=True &
CUDA_VISIBLE_DEVICES=1 python src/main.py --collect --config=qmix --env-config=sc2_v2_terran --offline_data_quality=expert --save_replay_buffer=True --num_episodes_collected=4000 --stop_winrate=0.9 --seed=1 --map_name=10gen_terran_5_vs_6 --use_wandb=True &
wait 
# smac上测试trsf
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="smac off" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=2 --use_wandb=True --wandb_note="smac off" &
wait
#* 无训练phigen性能正常
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=3 --lr=0.0005 --use_wandb=True --wandb_note="smac pre lr=0.0005" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=3 --lr=0.001 --use_wandb=True --wandb_note="smac pre lr=0.001" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=3 --lr=0.002 --use_wandb=True --wandb_note="smac pre lr=0.002" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=3 --lr=0.005 --use_wandb=True --wandb_note="smac pre lr=0.005" &
wait
#* 影响不大，很快会暴涨（若按照过拟合的理解则是很快已经解出，或是无需求解）
# 测试with pre phi
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="smac off w pre" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=2 --use_wandb=True --wandb_note="smac off w pre" &
wait
## smac off2on
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=sc2_transfer --task-config=sc2_test --seed=1 --use_wandb=True --wandb_note="smac off2on" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=sc2_transfer --task-config=sc2_test --seed=2 --use_wandb=True --wandb_note="smac off2on" &
wait
# -TODO 给trsf也适配下parallel? 先看看流程中哪些地方慢；但更大的问题是显存占用✅过了多次网络

# new arch: phimix
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="phimix pre" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="phimix pre" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="phimix pre a_loss dominate" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="phimix pre a_loss dominate" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="phimix pre thres w_reg fix" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="phimix pre thres w_reg fix" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=1234 --use_wandb=True --wandb_note="lbf pre" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=1234 --use_wandb=True --wandb_note="lbf pre weak lr sched" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre weak lr sched" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=1234 --use_wandb=True --wandb_note="lbf pre cos lr sched" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre cos lr sched" &
wait
#* 到目前位置1/T调度最有效

sleep 2h
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=1234 --use_wandb=True --wandb_note="lbf pre exp lr sched" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre exp lr sched" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre 1_T lr sched ms-fixed" &
wait
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre exp lr sched ms-fixed" &
wait
#* exp-lr不稳定

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre lin-decay lr sched" &
wait

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=2345 --use_wandb=True --wandb_note="lbf pre quad-decay lr sched"
#* 这些也行，但1/T确实收敛很快; linear也能降，解释不了用lin写
#* linear decay在3p2f上aloss降得更低, 可能是因为lr缩减太快，导致有点欠拟合
#- 0412先这样吧, 1/T快但是野路子，需要调，参数敏感；lin慢一点但loss更低，有支撑；
#* 1/T稳定，lin可能是过拟合，会炸

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=3407 --use_wandb=True --wandb_note="lbf pre"

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="lbf off const lr" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="lbf off const lr" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=1 --use_wandb=True --wandb_note="lbf off 1/T lr" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=2 --use_wandb=True --wandb_note="lbf off 1/T lr" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre modulate phi" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre modulate phi" &
wait
CUDA_VISIBLE_DEVICES=2 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 off relu" --checkpoint_path= --load_step=300005 &
CUDA_VISIBLE_DEVICES=3 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 off relu" --checkpoint_path= --load_step=300005 &
wait

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre no scale phi"
#* w_reg_Gate提高：应遵循某种够用原则，即使得w_reg总是处于Gate之下或者稳定在gate附近；更大的gate允许更宽范围的w，这对aloss下降也有帮助
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre no scale large gate" --save_model_interval=50000 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre no scale large gate" --save_model_interval=50000 &
wait
# xx关于off训练任务无性能，测试任务有性能的原因猜测：预训练阶段w值域受约束没有很好对齐奖励，sc2中的奖励存在较大值，这种对齐不足使得尽管loss很小，但实际上对稀疏而较大的奖励值预测不足，从而导出的psi以及策略较差。已学习任务读取错误对齐的w用来拟合奖励和Q函数偏差较大，未学习任务通过对已对齐的phi，基于固定均匀初始化的w进行组合仍然能导出对应的较好值函数与策略。xx
#* controller执行里面有个bug
#! mixing_n的问题，已修复

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 off relu fixed exec" --checkpoint_path=/home/amax/xyc/MATr/offpymarl/results/transfer/sc2/sc2_test/tr_sf/seed_4_tr_sf_2025-04-14_12-56-47/models/pretrain/ --load_step=100002 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 off relu fixed exec" --checkpoint_path=/home/amax/xyc/MATr/offpymarl/results/transfer/sc2/sc2_test/tr_sf/seed_4_tr_sf_2025-04-14_12-56-47/models/pretrain/ --load_step=100002 &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 off piecewise linear mixer" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 off piecewise linear mixer" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 off linear mixer" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 off linear mixer" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 off independent" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 off independent" &
wait
#* linear, leakyrelu 挺好; 
#* 0414 off 大约5～10k步收敛

# ablate phi_dim for sc2_test
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre phi 64" --phi_dim=64 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre phi 64" --phi_dim=64 &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre phi 32" --phi_dim=32 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre phi 32" --phi_dim=32 &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre phi 24" --phi_dim=24 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre phi 24" --phi_dim=24 &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre phi 16" --phi_dim=16 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre phi 16" --phi_dim=16 &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre phi 8" --phi_dim=8 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre phi 8" --phi_dim=8 &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=4 --use_wandb=True --wandb_note="sc2 pre phi 4" --phi_dim=4 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_transfer --task-config=sc2_test --seed=5 --use_wandb=True --wandb_note="sc2 pre phi 4" --phi_dim=4 &
wait
#* 仍然可以通过“冗余维度”来解释：较小的维度会让rloss或aloss一者更好而另一者较差，rloss更差时wreg也会更差，二更大的维度则使得三个loss都接近理想。

CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_v2_zerg --task-config=sc2_v2_test_zerg --seed=4 --use_wandb=True --wandb_note="sc2v2-zerg pre" &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=sc2_v2_zerg --task-config=sc2_v2_test_zerg --seed=5 --use_wandb=True --wandb_note="sc2v2-zerg pre" &
wait
#* 除开mixingn之外，w本身可能有一些问题，大量的正负相消是否也是一种过拟合？w只该表示重要性而不去进行正负翻转；实际上所有phi分量是有益的
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_v2_zerg --task-config=sc2_v2_test_zerg --seed=4 --use_wandb=True --wandb_note="sc2v2-zerg off w>0" --checkpoint_path=/home/amax/xyc/MATr/offpymarl/results/transfer/sc2_v2/sc2_v2_test/tr_sf/seed_4_tr_sf_2025-04-15_19-12-01/models/pretrain/ --load_step=200005 &
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=sc2_v2_zerg --task-config=sc2_v2_test_zerg --seed=5 --use_wandb=True --wandb_note="sc2v2-zerg off w>0" --checkpoint_path=/home/amax/xyc/MATr/offpymarl/results/transfer/sc2_v2/sc2_v2_test/tr_sf/seed_4_tr_sf_2025-04-15_19-12-01/models/pretrain/ --load_step=200005 &
wait
#* 已解决；更稳定了一点

# 开始调online
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf pre"

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf off" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf off" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf off lr-0.001" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf off lr-0.001" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf adapt" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf adapt" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf adapt no w_eff" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf adapt no w_eff" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf adapt mixing_w" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf adapt mixing_w" &
wait

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf adapt all-1" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf adapt all-1" &
wait
#* 都提升不了，性能差异不大，w上的细微差异并不能使策略有大幅改变；全1和mixningw更稳定; 可能是源于normed form所给的稠密w并不适合于直接微调w。
# 验证下是否是w表达能力的问题：设定Gate到很大
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf pre XL Gate" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf pre XL Gate" &
wait
#* 也许方差缩减性（同样起源于normed form）代替了w_reg? w_reg本身有控制学习误差的效果，可能对应倾向于保守的表征和激进的表征（值域范围明显扩大）；无限制时w->phi/r, phi的稳定性可能通过id以及与w互作用保持
#* 应该是normed form本身的性质，w和phi都能很好地收敛

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf off XL Gate" --checkpoint_path=/home/amax/xyc/MATr/offpymarl/results/transfer/lbforaging/lbf_test/tr_sf/seed_4_tr_sf_2025-04-16_14-32-55/models/pretrain/ --load_step=200005 &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf off XL Gate" --checkpoint_path=/home/amax/xyc/MATr/offpymarl/results/transfer/lbforaging/lbf_test/tr_sf/seed_4_tr_sf_2025-04-16_14-32-55/models/pretrain/ --load_step=200005 &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf adapt XL Gate" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_adapt --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf adapt XL Gate" &
wait
#* 没啥用

# test online (normal Gate)
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf on normal Gate" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf on normal Gate" &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf on less online mt" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf on less online mt" &
wait
#* 全架构调会不稳定
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf on no explore" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf on no explore" &
wait
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf on with phi" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_on --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf on with phi" &
wait

# added w as psi cond
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf pre only w_1"
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf pre for wcond psi"
CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_pre --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf pre std-r"

CUDA_VISIBLE_DEVICES=0 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=5 --use_wandb=True --wandb_note="lbf off w_1 wcond psi" &
CUDA_VISIBLE_DEVICES=1 python src/main.py --transfer --config=tr_sf_mto --env-config=gymma_transfer --task-config=lbf_test --seed=4 --use_wandb=True --wandb_note="lbf off w_1 wcond psi" &
wait

#* w cond 加在GRU前容易爆炸；可能加在GRU后MLP前: # TODO 参考USFA17页重新实现
