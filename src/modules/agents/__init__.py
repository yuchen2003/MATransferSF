REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .multi_task.mt_rnn_agent import MTRNNAgent
from .multi_task.odis_agent import ODISAgent
REGISTRY["mt_rnn"] = MTRNNAgent
REGISTRY["mt_odis"] = ODISAgent

from .transfer.tr_rnn_agent import TrRNNAgent
REGISTRY["tr_rnn"] = TrRNNAgent
