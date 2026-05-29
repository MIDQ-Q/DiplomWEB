import numpy as np
import logging

logger = logging.getLogger(__name__)

class BlockInterleaver:
    def __init__(self, depth: int = 8):
        if depth < 2: raise ValueError("depth must be >= 2")
        self.depth = depth

    def interleave(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(bits)) % self.depth
        if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        width = len(bits) // self.depth
        return bits.reshape(self.depth, width).T.ravel()

    def deinterleave(self, bits: np.ndarray, original_len: int) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(bits)) % self.depth
        if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        width = len(bits) // self.depth
        return bits.reshape(width, self.depth).T.ravel()[:original_len]

def calculate_rs_optimal_depth(max_burst_bits: int = 32, rs_nsym: int = 10) -> int:
    t = max(1, rs_nsym // 2)
    return max(2, int(np.ceil(max_burst_bits / (8.0 * t))))

def get_interleaver(config: dict):
    cfg = config.get("interleaving", {})
    if not cfg.get("enabled", False): return None
    depth = cfg.get("depth")
    coding = config.get("coding", {})
    if depth is None and coding.get("enabled") and coding.get("type") in ("reed-solomon", "rs"):
        burst = cfg.get("expected_burst_bits", 32)
        nsym = coding.get("rs_nsym", 10)
        depth = calculate_rs_optimal_depth(burst, nsym)
        logger.info(f"[Interleaver] Auto depth for RS: {depth} (burst={burst}b, nsym={nsym})")
    elif depth is None:
        depth = 8
    return BlockInterleaver(depth=int(depth))