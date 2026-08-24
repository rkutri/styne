import numpy as np
import tracemalloc
import time
from styne.gp.gaussianprocess import GaussianProcess
from styne.statistics.interface import CovarianceFunctionInterface
from styne.utility.grid import UniformGrid

class MockCov(CovarianceFunctionInterface):
    def evaluate_covariance(self, x1, x2): return np.zeros((len(x1), len(x2)))
    def evaluate_fourier(self, k): return np.ones(k.shape[0])

def audit_evaluation(q=50, steps=100):
    gp = GaussianProcess.dna(MockCov(), q, d=1)
    sites = UniformGrid(0.0, 1.0, 100)
    gp.sites = sites
    
    rng = np.random.default_rng(42)
    thetas = rng.standard_normal((steps, gp.parameterDimension))
    
    # Warmup
    _ = gp.at_sites(thetas[0])
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    # Snapshot at start
    snapshot1 = tracemalloc.take_snapshot()
    
    for i in range(steps):
        _ = gp.at_sites(thetas[i])
        
    duration = time.perf_counter() - start_time
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print(f"\nAudit for q={q}, steps={steps}:")
    print(f"  Total time: {duration:.4f}s ({duration/steps:.6f}s per eval)")
    
    # Filter for interesting entries in our repo
    total_size = 0
    total_count = 0
    for stat in stats:
        if 'nelo' in stat.traceback[0].filename and '__pycache__' not in stat.traceback[0].filename:
            # Note: diff stays positive if we allocate more than we free
            # But tracemalloc shows total current alive blocks.
            # To see *total* allocations, we'd need a different tool, but we can look at 
            # size_diff if the GC hasn't caught up.
            pass
            
    # Actually, let's just use the 'peak' memory or total blocks
    # Better yet, let's use a simple counter if we can, or just time it.
    # Python GC makes tracemalloc tricky for "total allocations ever".
    # But timing + manual audit of code is good.

if __name__ == "__main__":
    audit_evaluation(q=50, steps=1000)
    audit_evaluation(q=200, steps=1000)
