"""Packed INT4 weights + scalar-semantics layer kernel (NumPy). Mirrors cpp/include/laminar/int4.h."""
import numpy as np
from .config import UNIT_BYTES, UNIT_NIBBLES


def unpack_unit(unit_bytes):
    """uint8[2816] -> int8[5632] (nibble 2k = low nibble of byte k)."""
    lo = (unit_bytes & 15).astype(np.int16)
    hi = (unit_bytes >> 4).astype(np.int16)
    w = np.empty(UNIT_NIBBLES, dtype=np.int16)
    w[0::2] = lo
    w[1::2] = hi
    return (((w ^ 8) - 8)).astype(np.int8)


def pack_unit(w_int):
    """int8[5632] in [-8,7] -> uint8[2816]."""
    v = (w_int.astype(np.int16) & 15).astype(np.uint8)
    return (v[0::2] | (v[1::2] << 4)).astype(np.uint8)


def layer_forward(w_unit_int8, nib_off, n_in, n_out, shift, x):
    """out[j] = clamp((sum_i x[i] * W[j][i]) >> shift, -127, 127) with W row-major [out][in]."""
    W = w_unit_int8[nib_off:nib_off + n_in * n_out].reshape(n_out, n_in).astype(np.int32)
    acc = W @ x.astype(np.int32)
    return np.clip(acc >> shift, -127, 127).astype(np.int8)
