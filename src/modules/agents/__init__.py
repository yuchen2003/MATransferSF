REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent

from .multi_task.mt_rnn_agent import MTRNNAgent
from .multi_task.odis_agent import ODISAgent
REGISTRY["mt_rnn"] = MTRNNAgent
REGISTRY["mt_odis"] = ODISAgent

from .transfer.tr_sf_rnn_agent import TrSFRNNAgent
REGISTRY["tr_sf_rnn"] = TrSFRNNAgent

from .multi_task.hissd_agent import HISSDAgent
REGISTRY["mt_hissd"] = HISSDAgent
