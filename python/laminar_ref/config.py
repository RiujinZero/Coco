"""Unit geometry and parameters, mirroring cpp/include/laminar/config.h."""
from dataclasses import dataclass, field

IN, A, B, C, E, F, R2, R6 = 128, 16, 24, 16, 16, 32, 16, 8
R = R2 + R6
IN2, IN3, IN5, IN6, IN4 = A + R2, A + B, B + C, E + R6, F
P1, P2, P3, P5, P6, P4 = IN * A, IN2 * B, IN3 * C, IN5 * E, IN6 * F, IN4 * R
NIB_L1 = 0
NIB_L2 = NIB_L1 + P1
NIB_L3 = NIB_L2 + P2
NIB_L5 = NIB_L3 + P3
NIB_L6 = NIB_L5 + P5
NIB_L4 = NIB_L6 + P6
UNIT_NIBBLES = NIB_L4 + P4          # 5632
UNIT_BYTES = UNIT_NIBBLES // 2      # 2816
SENSORY_BITS = 8
TUMBLER_BITS = 56
SENSORY_MASK = 0xFF
LI_1, LI_2, LI_3, LI_5, LI_6, LI_4 = range(6)
# (nib_off, in, out) per layer id
LAYERS = [(NIB_L1, IN, A), (NIB_L2, IN2, B), (NIB_L3, IN3, C), (NIB_L5, IN5, E), (NIB_L6, IN6, F), (NIB_L4, IN4, R)]


@dataclass
class Params:
    units: int = 4096
    motif_size: int = 128
    k_set: tuple = (2, 4, 8, 16, 32, 64)
    max_hops: int = 2
    vocab: int = 32
    seed: int = 1
    shifts: tuple = (1, 2, 2, 3, 3, 3)
    death_l1: int = 6
    death_l23: int = 6
    death_l5: int = 6
    death_l6: int = 6
    death_rel: int = 0
    recirc_thresh: int = 40
    resist_min: int = 24
    resist_max: int = 255
    resist_init: int = 64
    resist_up: int = 2
    resist_up_dead: int = 1
    resist_down: int = 3
    novel_hits: int = 64
    surprise_thresh: int = 16384
    novelty_num: int = 1
    novelty_den: int = 2
    leaf_max: int = 4
    assoc_extra_bits: int = 1
    assoc_tighten_after: int = 16
    assoc_loosen_after: int = 64
    train: bool = True
    structural: bool = True
    hidden: str = "none"            # "none" | "perturb"
    lr_num: int = 1
    lr_shift: int = 9
    quantum: int = 64
    half_quantum: int = 32
    err_gain: float = 8.0
    lr_out: float = 0.005
    err_sparse: float = 0.01
    lr_anneal: int = 3
    perturb_delta: int = 8
    perturb_gain: int = 64
    hidden_shift: int = 0
    future_buckets: int = 32
    evidence_slots: int = 4096
    conflict_thresh: int = 22000
    split_min_evidence: int = 64
    split_g_per_df: float = 3.0
    lock_death_blocks: int = 0
    block: int = 4096
    max_active: int = 8192
    hist_bytes: int = 4
    ws_decay: int = 128
    ws_key: bool = False
    ws_input: bool = True
    l4_key: bool = False
