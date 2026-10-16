"""Streaming resampler: a microphone that only supports 44.1 / 48 kHz still has to feed 16 kHz mono frames to the wake word
(openWakeWord) and Whisper. Feeding them 48 kHz audio labelled as 16 kHz is the same speech slowed 3x and a third of its pitch — no error, just a
"silent" wake word — so the rate is converted here, with an anti-alias filter, one fixed-size frame at a time.

Block-wise polyphase resampling with context on both sides, so frame boundaries are seamless. The cost is latency of one or
two frames (30-60 ms), which a wake word does not notice.
"""
from __future__ import annotations

from math import gcd
from typing import List, Optional

import numpy as np


class UnsupportedRate(ValueError):
    """The device rate cannot be converted into whole `frame_out`-sample frames with this resampler."""


class StreamResampler:
    """Feed `block_in` samples at `src_rate`; get back `frame_out` samples at `dst_rate`, aligned and continuous."""

    _ZERO_CROSSINGS = 5

    def __init__(self, src_rate: int, dst_rate: int, frame_out: int):
        from scipy import signal

        self.src_rate, self.dst_rate, self.frame_out = int(src_rate), int(dst_rate), int(frame_out)
        g = gcd(self.src_rate, self.dst_rate)
        self.up, self.down = self.dst_rate // g, self.src_rate // g
        block, rem = divmod(self.frame_out * self.down, self.up)
        if rem or block % self.down:
            raise UnsupportedRate(f"cannot convert {self.src_rate} Hz to {self.dst_rate} Hz in {self.frame_out}-sample frames")
        self.block_in = block
        half = self._ZERO_CROSSINGS * max(self.up, self.down)
        self._h = signal.firwin(2 * half + 1, 1.0 / max(self.up, self.down), window=("kaiser", 5.0))        # resample_poly applies the x`up` gain itself
        self._signal = signal
        blocks_right = max(1, -(-half // self.block_in))                  # whole blocks of look-ahead
        left = -(-half // self.block_in) * self.block_in
        self._left = left - left % self.down if left % self.down else left
        self._right_blocks = blocks_right
        self._hist = np.zeros(self._left, dtype=np.float32)               # samples BEFORE the block being emitted
        self._pending: List[np.ndarray] = []                              # the block being emitted + its look-ahead

    @property
    def latency_frames(self) -> int:
        return self._right_blocks

    def push(self, block: np.ndarray) -> Optional[np.ndarray]:
        """Add one input block; returns one output frame (int16) once enough look-ahead has arrived, else None."""
        x = np.asarray(block, dtype=np.float32)
        if len(x) != self.block_in:
            raise ValueError(f"expected {self.block_in} input samples, got {len(x)}")
        self._pending.append(x)
        if len(self._pending) < self._right_blocks + 1:
            return None
        current = self._pending[0]
        window = np.concatenate([self._hist] + self._pending)
        y = self._signal.resample_poly(window, self.up, self.down, window=self._h)
        start = len(self._hist) * self.up // self.down
        out = y[start:start + self.frame_out]
        # slide: the emitted block becomes history
        self._hist = np.concatenate([self._hist, current])[-self._left:] if self._left else self._hist
        self._pending.pop(0)
        return np.clip(np.rint(out), -32768, 32767).astype(np.int16)
