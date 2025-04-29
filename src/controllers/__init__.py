REGISTRY = {}

from .basic_controller import BasicMAC
from .random_controller import RandomMAC
from .maddpg_controller import MADDPGMAC
from .n_controller import NMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY['random_mac'] = RandomMAC
REGISTRY['maddpg_mac'] = MADDPGMAC
REGISTRY["n_mac"] = NMAC

from .multi_task.mt_basic_controller import MTBasicMAC
from .multi_task.mt_maddpg_controller import MTMADDPGMAC
from .multi_task.mt_odis_controller import ODISMAC

REGISTRY["mt_basic_mac"] = MTBasicMAC
REGISTRY['mt_maddpg_mac'] = MTMADDPGMAC
REGISTRY["mt_odis_mac"] = ODISMAC

from .transfer.tr_basic_controller import TrBasicMAC
REGISTRY['tr_basic_mac'] = TrBasicMAC

from .multi_task.mt_hissd_controller import HISSDSMAC
REGISTRY["mt_hissd_mac"] = HISSDSMAC

from .multi_task.mt_updet_controller import UPDeTMAC
from .multi_task.mt_bc_controller import BCMAC
from .multi_task.mt_bcr_controller import BCRMAC
REGISTRY["mt_updet_mac"] = UPDeTMAC
REGISTRY["mt_bc_mac"] = BCMAC
REGISTRY["mt_bcr_mac"] = BCRMAC
