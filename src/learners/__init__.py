from .q_learner import QLearner
from .nq_learner import NQLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .null_learner import NullLearner
from .bc_learner import BCLearner
from .maddpg_learner import MADDPGLearner
from .matd3_learner import MATD3Learner
from .icq_learner import ICQLearner
from .itd3_learner import ITD3Learner
from .actor_critic_learner import ActorCriticLearner

REGISTRY = {}

REGISTRY["bc_learner"] = BCLearner
REGISTRY["q_learner"] = QLearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["null_learner"] = NullLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["matd3_learner"] = MATD3Learner
REGISTRY["icq_learner"] = ICQLearner
REGISTRY["itd3_learner"] = ITD3Learner
REGISTRY["actor_critic_learner"] = ActorCriticLearner

from .multi_task.mt_q_learner import MTQLearner
from .multi_task.mt_matd3_learner import MTMATD3Learner
from .multi_task.mt_bc_learner import MTBCLearner
from .multi_task.odis_learner import ODISLearner

REGISTRY["mt_q_learner"] = MTQLearner
REGISTRY["mt_matd3_learner"] = MTMATD3Learner
REGISTRY["mt_bc_learner"] = MTBCLearner
REGISTRY["odis_learner"] = ODISLearner

from .transfer.tr_sf_learner import TransferSFLearner
from .transfer.tr_q_learner import TransferQLearner
REGISTRY["tr_sf_learner"] = TransferSFLearner
REGISTRY["tr_q_learner"] = TransferQLearner
