import numpy as np
import time
from scipy.fft import dct

def baseline_series(a, q):
    acopy = np.array(a / np.sqrt(2.))
    acopy[0] *= np.sqrt(2.)
    # Allocation 1 (pad)
    acopy_padded = np.pad(acopy, (0, 1))
    # Allocation 2 (dct)
    return dct(acopy_padded, norm="backward", type=1)

def buffered_series(a, q, pad_buffer):
    # a is length q+1. pad_buffer is length q+2.
    # Assignment into slice (no allocation)
    np.multiply(a, 1.0/np.sqrt(2.0), out=pad_buffer[:-1])
    pad_buffer[0] *= np.sqrt(2.0)
    pad_buffer[-1] = 0.0
    # dct with overwrite_x=True (might avoid allocation)
    return dct(pad_buffer, norm="backward", type=1, overwrite_x=True)

def test_speedup(q=200, steps=10000):
    a = np.random.randn(q + 1)
    pad_buffer = np.zeros(q + 2)
    
    # Warmup
    _ = baseline_series(a, q)
    _ = buffered_series(a, q, pad_buffer)
    
    start = time.perf_counter()
    for _ in range(steps):
        _ = baseline_series(a, q)
    t_base = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(steps):
        _ = buffered_series(a, q, pad_buffer)
    t_buf = time.perf_counter() - start
    
    print(f"q={q}, steps={steps}")
    print(f"  Baseline: {t_base:.4f}s")
    print(f"  Buffered: {t_buf:.4f}s")
    print(f"  Speedup: {t_base/t_buf:.2f}x")

if __name__ == "__main__":
    test_speedup(q=50)
    test_speedup(q=200)
    test_speedup(q=500)
