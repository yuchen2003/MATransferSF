import numpy as np

def polynomial_embed(v, length, v_min, v_max, exp=0.5):
    rel_v = (v - v_min) / (v_max - v_min)
    n_filled = round(rel_v**exp * length)
    embed_vec = np.zeros(length,)
    embed_vec[:n_filled] = 1
    return embed_vec

def binary_embed(v, length, v_max):
    assert 2**length - 1 >= v_max
    embed_vec = np.zeros(length,)
    bin_v = [int(item) for item in list(bin(v)[2:])]
    embed_vec[-len(bin_v):] = bin_v
    return embed_vec
    
def onehot_embed(v, length):
    assert 0 <= v < length
    embed_vec = np.zeros(length,)
    embed_vec[v] = 1.
    return embed_vec

def pad_shape(z, x, extra=0, pos=0):
    ''' assume len(x.shape) > len(z.shape). E.g.
        x: [bs1, x, y, z, da]
        z: [bs2, db]
        pos: padding shape start from z.shape[pos]
        return: z in [bs2, db, 1, 1, 1; extra 1's] view
    '''
    return z.view(*z.shape[:pos], *([1] * (len(x.shape) - len(z.shape) + extra)), *z.shape[pos:])
