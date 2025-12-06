"""
Lightweight local shim for the `tenacity` API used by tests and modules.

This is intentionally minimal: it provides no real retry semantics but
exposes the common names used across the codebase so tests can import
`tenacity` even if the package is not installed in the environment.

If you have `tenacity` installed, prefer removing this shim so the
real implementation is used.
"""
from functools import wraps


def retry(*dargs, **dkwargs):
    # acts as a no-op decorator when real tenacity isn't available
    def _decorator(fn):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return _wrapped

    # support usage as @retry or @retry(...) by returning decorator
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return _decorator(dargs[0])
    return _decorator


def stop_after_attempt(n):
    class _StopAfter:
        def __init__(self, n):
            self.n = n

        def __repr__(self):
            return f"stop_after_attempt({self.n})"

    return _StopAfter(n)


def wait_exponential(multiplier=1, max=None, exp_base=2, min=None):
    # accept both `min` and `max` kwarg names as callers in this repo use `min`
    class _Wait:
        def __init__(self, multiplier, min_val, max_val, exp_base):
            self.multiplier = multiplier
            self.min = min_val
            self.max = max_val
            self.exp_base = exp_base

        def __repr__(self):
            return f"wait_exponential(multiplier={self.multiplier}, min={self.min}, max={self.max}, exp_base={self.exp_base})"

    return _Wait(multiplier, min, max, exp_base)


def retry_if_exception_type(exc_type):
    # returns a predicate that can be used similarly to tenacity
    def _predicate(exc):
        return isinstance(exc, exc_type)

    return _predicate


class Retrying:
    # Minimal placeholder for code that instantiates Retrying
    def __init__(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter(())
