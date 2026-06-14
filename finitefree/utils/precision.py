from typing import Any, Optional

import flint


class PrecisionContext:
    def __init__(self, degree: int, prec: Optional[int] = None) -> None:
        self.degree = degree
        self.original_prec = flint.ctx.prec
        if prec is not None:
            self.new_prec = prec
        else:
            # Dynamically scale precision based on degree to handle combinatorial explosions
            self.new_prec = max(53, int(degree * 2.5))

    def __enter__(self) -> "PrecisionContext":
        flint.ctx.prec = self.new_prec
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        flint.ctx.prec = self.original_prec
