import numpy as np
import pytest
from scipy.fft import dct, dst
from styne.gp.dnautility import (
    cos_series, sin_series, 
    adj_cos_series_1d, adj_sin_series_1d,
    BC, BoundaryCondition
)
from styne.gp.dna import (
    DNAFourierComponentRealisation,
    DNAFourierRealisation,
    DNAFourierEngine
)
from styne.utility.grid import UniformGrid

def test_series_1d_batch():
    """Verify that 1D series handle 2D inputs (batches) correctly."""
    q = 10
    nBatch = 5
    rng = np.random.default_rng(42)
    
    # Cosine
    a = rng.standard_normal((nBatch, q + 1))
    # Batch call
    resBatch = cos_series(a, axis=-1)
    
    # Loop call
    resLoop = []
    for i in range(nBatch):
        resLoop.append(cos_series(a[i]))
    resLoop = np.array(resLoop)
    
    assert np.allclose(resBatch, resLoop, rtol=1e-13, atol=1e-13)
    
    # Sine
    b = rng.standard_normal((nBatch, q + 1))
    b[:, 0] = 0 # sin_series expects a[0]=0
    resBatch = sin_series(b, axis=-1)
    
    resLoop = []
    for i in range(nBatch):
        resLoop.append(sin_series(b[i]))
    resLoop = np.array(resLoop)
    
    assert np.allclose(resBatch, resLoop, rtol=1e-13, atol=1e-13)

def test_adjoint_series_1d_batch():
    """Verify that adjoint 1D series handle 2D inputs (batches) correctly."""
    q = 10
    nBatch = 5
    nG = q + 2
    rng = np.random.default_rng(43)
    
    # r check
    r = rng.standard_normal((nG, nBatch)) # adjoint axis=0 is grid
    
    # Adjoint Cosine
    resBatch = adj_cos_series_1d(r, q, axis=0)
    
    resLoop = []
    for i in range(nBatch):
        resLoop.append(adj_cos_series_1d(r[:, i], q))
    # Loop returns vectors of size q+1
    # We expect resBatch to be (q+1, nBatch)
    resLoop = np.array(resLoop).T
    
    assert np.allclose(resBatch, resLoop, rtol=1e-13, atol=1e-13)
    
    # Adjoint Sine
    resBatch = adj_sin_series_1d(r, q, axis=0)
    resLoop = []
    for i in range(nBatch):
        resLoop.append(adj_sin_series_1d(r[:, i], q))
    resLoop = np.array(resLoop).T
    
    assert np.allclose(resBatch, resLoop, rtol=1e-13, atol=1e-13)

def test_dna_realisation_batch_1d():
    """Verify DNAFourierRealisation handles batch evaluation in 1D."""
    q = 10
    nBatch = 5
    nG = q + 2
    rng = np.random.default_rng(44)
    
    real = DNAFourierRealisation(q, d=1)
    nParam = real.dimension
    
    # 1. Single sample check
    theta = rng.standard_normal(nParam)
    real.coefficient = theta
    u_single = real.evaluate_native()
    assert u_single.shape == (nG,)
    
    # 2. Batch sample check
    thetas = rng.standard_normal((nBatch, nParam))
    real.coefficient = thetas
    u_batch = real.evaluate_native()
    assert u_batch.shape == (nBatch, nG)
    
    # Verify correspondence
    for i in range(nBatch):
        real.coefficient = thetas[i]
        u_truth = real.evaluate_native()
        assert np.allclose(u_batch[i], u_truth, rtol=1e-13, atol=1e-13)
    
    print("DNA Realisation 1D Batch PASSED")

def test_adjoint_inner_product_consistency():
    """Rigorous check: <Jv, w> == <v, J^T w> for DNA engine."""
    q = 10
    d = 1
    engine = DNAFourierEngine(q, d)
    # Use non-matching sites to test interpolation part too
    sites = UniformGrid(0.1, 0.9, 15)
    engine.set_sites(sites)
    
    real = engine.build_realisation()
    nParam = real.dimension
    nSites = len(sites)
    
    rng = np.random.default_rng(45)
    v = rng.standard_normal(nParam)
    w = rng.standard_normal(nSites)
    
    # Jv = engine.apply_jacobian(v, None)
    # JTw = engine.apply_adjoint_jacobian(w, None)
    # The above methods are what apply_jacobian and apply_adjoint_jacobian do.
    
    # We verify <Jv, w> = <v, J^T w>
    Jv = engine.apply_jacobian(v, None)
    JTw = engine.apply_adjoint_jacobian(w, None)
    
    lhs = np.dot(Jv, w)
    rhs = np.dot(v, JTw)
    
    print(f"  <Jv, w>: {lhs:.12f}")
    print(f"  <v, JTw>: {rhs:.12f}")
    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)
    print("Adjoint Inner-Product Consistency PASSED")

def test_engine_at_sites_batch():
    """Verify DNAFourierEngine.at_sites handles batches."""
    q = 10
    d = 1
    engine = DNAFourierEngine(q, d)
    sites = UniformGrid(0.1, 0.9, 8)
    engine.set_sites(sites)
    
    real = engine.build_realisation()
    nBatch = 4
    rng = np.random.default_rng(46)
    thetas = rng.standard_normal((nBatch, real.dimension))
    
    real.coefficient = thetas
    u_batch = engine.at_sites(real, sites)
    assert u_batch.shape == (nBatch, len(sites))
    
    # Loop verify
    for i in range(nBatch):
        real.coefficient = thetas[i]
        u_truth = engine.at_sites(real, sites)
        assert np.allclose(u_batch[i], u_truth, rtol=1e-13, atol=1e-13)
    
    print("DNA Engine at_sites Batch PASSED")

if __name__ == "__main__":
    test_series_1d_batch()
    test_adjoint_series_1d_batch()
    test_dna_realisation_batch_1d()
    test_adjoint_inner_product_consistency()
    test_engine_at_sites_batch()
    print("\nALL VECTORIZATION TESTS PASSED")
