from typing import Any, List, Optional

import flint


class PrecisionContext:
    """Temporarily set process-wide FLINT working precision in bits.

    The default is max(53, int(2.5*degree)); prec overrides it. Each entry
    records the precision current at that entry, and the matching exit restores
    it, including on exceptions. One object may be reused sequentially or
    nested. This is not an accuracy certificate, a memory manager or a
    thread-local context.
    """

    def __init__(self, degree: int, prec: Optional[int] = None) -> None:
        self.degree = degree
        # Precision saved by the most recent entry (construction value before).
        self.original_prec = flint.ctx.prec
        self._saved_precs: List[int] = []
        if prec is not None:
            self.new_prec = prec
        else:
            # Dynamically scale precision based on degree to handle combinatorial explosions
            self.new_prec = max(53, int(degree * 2.5))

    def __enter__(self) -> "PrecisionContext":
        saved = flint.ctx.prec
        self._saved_precs.append(saved)
        self.original_prec = saved
        flint.ctx.prec = self.new_prec
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        flint.ctx.prec = self._saved_precs.pop()
        if self._saved_precs:
            self.original_prec = self._saved_precs[-1]
