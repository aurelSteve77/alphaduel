"""Singleton metaclass."""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar


class Singleton(type):
    """Metaclass that ensures only one instance of a class exists.

    Thread-safe: concurrent first constructions are serialized with a lock.
    Subsequent calls return the same instance (constructor args are ignored).
    """

    _instances: ClassVar[dict[type, Any]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
