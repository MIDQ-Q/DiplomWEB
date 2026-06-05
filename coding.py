# coding.py
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import reedsolo
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

class HammingCoder:
    def __init__(self, n: int = 7, k: int = 4):
        # Валидация стандартных кодов Хэмминга
        valid = {(7,4):3, (15,11):4, (31,26):5, (63,57):6, (127,120):7}
        if (n,k) not in valid:
            raise ValueError(f"Hamming: допустимы только {(list(valid.keys()))}, получено ({n},{k})")
        self.n, self.k = n, k
        self.rate = k/n
        self.name = f"Hamming({n},{k})"
        self._G, self._H, self._syn_table = self._build()

    def _build(self):
        r = int(np.log2(self.n+1))
        pp = [2**i - 1 for i in range(r)]
        dp = [j for j in range(self.n) if j not in pp]
        H = np.zeros((r, self.n), dtype=np.uint8)
        for j in range(self.n):
            for i in range(r): H[i, j] = (j+1)>>i & 1
        G = np.zeros((self.k, self.n), dtype=np.uint8)
        for row, d in enumerate(dp):
            G[row, d] = 1
            for i, p in enumerate(pp): G[row, p] = (d+1)>>i & 1

        # ✅ Исправлено: явное преобразование в Python-список перед join()
        table = {}
        for j in range(self.n):
            e = np.zeros(self.n, dtype=np.uint8)
            e[j] = 1
            syndrome = ((H @ e) % 2).tolist()
            table[int("".join(str(int(x)) for x in syndrome), 2)] = j
        return G, H, table

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(bits))%self.k
        if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return (bits.reshape(-1, self.k) @ self._G % 2).ravel()

    def decode(self, bits: np.ndarray) -> Tuple[np.ndarray, dict]:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(bits))%self.n
        if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        blocks = bits.reshape(-1, self.n)
        corr, det = 0, 0
        dec = np.empty((len(blocks), self.k), dtype=np.uint8)
        dp = [j for j in range(self.n) if j not in [2**i-1 for i in range(int(np.log2(self.n+1)))]]

        for i, b in enumerate(blocks):
            # ✅ Исправлено: безопасное преобразование синдрома в бинарную строку
            syndrome = ((self._H @ b) % 2).tolist()
            s = int("".join(str(int(x)) for x in syndrome), 2)

            if s!=0:
                if s in self._syn_table:
                    b = b.copy()
                    b[self._syn_table[s]] ^= 1
                    corr += 1
                else:
                    det += 1
            dec[i] = b[dp]
        return dec.ravel(), {"corrected": corr, "detected": det, "blocks": len(blocks)}

class ReedSolomonCoder:
    def __init__(self, nsym: int = 10, nsize: int = 255, fcr: int = 0, prim: int = 0x11d):
        if not _RS_AVAILABLE: raise RuntimeError("reedsolo не установлен")
        if nsize > 255: nsize = 255
        if nsym >= nsize: raise ValueError(f"nsym={nsym} должен быть < nsize={nsize}")
        self.nsym, self.nsize, self.k = nsym, nsize, nsize-nsym
        self.rate = self.k/nsize
        self.name = f"Reed-Solomon(nsym={nsym})"
        self._codec = reedsolo.RSCodec(nsym=nsym, nsize=nsize, fcr=fcr, prim=prim)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(bits))%8
        if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        data = np.packbits(bits).tobytes()[:len(bits)//8]
        out = bytearray()
        for i in range(0, len(data), self.k):
            out.extend(self._codec.encode(data[i:i+self.k]))
        return np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))

    def decode(self, bits: np.ndarray) -> Tuple[np.ndarray, dict]:
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-len(bits))%8
        if pad: bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        raw = np.packbits(bits).tobytes()
        dec, corr, det, total = bytearray(), 0, 0, 0
        for i in range(0, len(raw), self.nsize):
            block = raw[i:i+self.nsize]
            if len(block)<self.nsize: block += bytes(self.nsize-len(block))
            total += 1
            try:
                d, _, err = self._codec.decode(block)
                dec.extend(bytes(d))
                if err: corr += len(err)
            except reedsolo.ReedSolomonError:
                det += 1; dec.extend(block[:self.k])
        res = np.unpackbits(np.frombuffer(bytes(dec), dtype=np.uint8))
        return res, {"corrected": corr, "detected": det, "blocks": total}

def create_coder(cfg: dict) -> Tuple[Optional[object], float]:
    if not cfg.get("enabled", False): return None, 1.0
    t = cfg.get("type", "hamming").lower()
    if t == "hamming":
        c = HammingCoder(n=int(cfg.get("n",7)), k=int(cfg.get("k",4)))
    elif t in ("reed-solomon", "rs"):
        c = ReedSolomonCoder(
            nsym=int(cfg.get("rs_nsym",10)), nsize=int(cfg.get("rs_nsize",255)),
            fcr=int(cfg.get("rs_fcr",0)), prim=int(cfg.get("rs_prim","0x11d"),16)
        )
    else: raise ValueError(f"Неизвестный кодек: {t}")
    return c, c.rate

def rs_available() -> bool:
    return _RS_AVAILABLE