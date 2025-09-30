import time
from functools import wraps

def log_action(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[LOG] {func.__name__} called")
        return func(*args, **kwargs)
    return wrapper

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = func(*args, **kwargs)
        ms = (time.time() - t0) * 1000
        if isinstance(out, dict):
            out["_ms"] = ms
        return out
    return wrapper
