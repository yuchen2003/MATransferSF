from .odis_learner import ODISLearner
from .updet_learner import UPDeTLearner
from .bc_learner import BCLearner
from .hissd_learner import HISSDLearner

REGISTRY = {}

REGISTRY["odis_learner"] = ODISLearner
REGISTRY["updet_learner"] = UPDeTLearner
REGISTRY["bc_learner"] = BCLearner
REGISTRY["hissd_learner"] = HISSDLearner
