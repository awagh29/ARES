"""MATLAB ``RandStream('mt19937ar','Seed',s)`` draws, so outage schedules match MATLAB.

MT19937 with the 2002 ``init_genrand`` seeding and 53-bit ``genrand_res53``
doubles. The same generator appears in ``ros2/.../matlab_rand.py``; this copy
keeps the test pipeline self-contained. ``randi`` uses ``floor(imax*rand)+1``;
the parity tests confirm that against MATLAB.
"""

from __future__ import annotations

import math


class MatlabMT19937:
    def __init__(self, seed: int):
        seed = int(seed)
        if seed == 0:
            seed = 5489  # MATLAB maps seed 0 to the reference MT19937 default seed
        self.mt = [0] * 624
        self.mt[0] = seed & 0xFFFFFFFF
        for index in range(1, 624):
            previous = self.mt[index - 1]
            self.mt[index] = (1812433253 * (previous ^ (previous >> 30)) + index) & 0xFFFFFFFF
        self.index = 624

    def _twist(self) -> None:
        mt = self.mt
        for index in range(624):
            mixed = (mt[index] & 0x80000000) + (mt[(index + 1) % 624] & 0x7FFFFFFF)
            twisted = mt[(index + 397) % 624] ^ (mixed >> 1)
            if mixed & 1:
                twisted ^= 0x9908B0DF
            mt[index] = twisted & 0xFFFFFFFF
        self.index = 0

    def _uint32(self) -> int:
        if self.index >= 624:
            self._twist()
        value = self.mt[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0xFFFFFFFF

    def rand(self) -> float:
        high = self._uint32() >> 5
        low = self._uint32() >> 6
        return (high * 67108864.0 + low) * (1.0 / 9007199254740992.0)

    def randi(self, imax: int) -> int:
        return int(math.floor(imax * self.rand())) + 1
