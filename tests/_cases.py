"""Keep batched contract failures attributable to their original input cases."""

from contextlib import contextmanager
from typing import Any, Iterator

import pytest


@contextmanager
def case(**parameters: Any) -> Iterator[None]:
    """Label a failed scenario without swallowing skips or process interrupts."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        label = ", ".join(f"{name}={value!r}" for name, value in parameters.items())
        raise AssertionError(f"Contract scenario ({label}) failed: {error}") from error
