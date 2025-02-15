REGISTRY = {}

from .sc2_decomposer import SC2Decomposer
REGISTRY["sc2_decomposer"] = SC2Decomposer

from .cn_decomposer import MPEDecomposer
REGISTRY["mpe_decomposer"] = MPEDecomposer

from .lbf_decomposer import GymDecomposer
REGISTRY["gymma_decomposer"] = GymDecomposer
