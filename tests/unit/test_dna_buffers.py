import numpy as np
import pytest
from styne.gp.dna import DNAFourierRealisation, DNAFourierComponentRealisation
from styne.gp.dnautility import BC, BoundaryCondition

def test_component_buffer_reuse():
    """Verify that DNAFourierComponentRealisation reuses its scratch buffer."""
    q = 50
    bc = (BC.NEUMANN,)
    comp = DNAFourierComponentRealisation(BoundaryCondition(bc), q)
    
    rng = np.random.default_rng(123)
    theta1 = rng.standard_normal(comp.dimension)
    
    comp.project(theta1)
    res1 = comp.evaluate_native().copy()
    buf_id1 = id(comp._scratch)
    assert comp._scratch is not None
    
    # Evaluate again with different params
    theta2 = rng.standard_normal(comp.dimension)
    comp.project(theta2)
    res2 = comp.evaluate_native()
    buf_id2 = id(comp._scratch)
    
    assert buf_id1 == buf_id2
    assert not np.allclose(res1, res2)
    print("Component buffer reuse PASSED")

def test_realisation_buffer_reuse():
    """Verify that DNAFourierRealisation reuses its native grid buffer."""
    q = 20
    d = 1
    real = DNAFourierRealisation(q, d)
    
    rng = np.random.default_rng(124)
    theta1 = rng.standard_normal(real.dimension)
    
    real.coefficient = theta1
    res1 = real.evaluate_native()
    buf_id1 = id(real._nativeBuffer)
    
    real.coefficient = rng.standard_normal(real.dimension)
    res2 = real.evaluate_native()
    buf_id2 = id(real._nativeBuffer)
    
    assert buf_id1 == buf_id2
    print("Realisation buffer reuse PASSED")

def test_buffer_numerical_consistency():
    """Verify that buffered results are numerically identical to expected values."""
    q = 10
    d = 1
    real = DNAFourierRealisation(q, d)
    rng = np.random.default_rng(125)
    theta = rng.standard_normal(real.dimension)
    
    real.coefficient = theta
    res = real.evaluate_native()
    
    # Manually compute one BC block to compare
    # Block 0 is Neumann
    comp = real._param.block(0).function
    u_comp = comp.evaluate_native()
    
    # The scale is 2**(-0.5)
    scale = 2.**(-0.5)
    # The full field is the sum of blocks * scale
    u_comp_total = 0
    for i in range(real._param.nBlocks):
        u_comp_total += real._param.block(i).function.evaluate_native()
    u_comp_total *= scale
    
    assert np.allclose(res, u_comp_total, rtol=1e-15, atol=1e-15)
    print("Buffer numerical consistency PASSED")

def test_clone_isolation():
    """Verify that cloned realisations do not share buffers."""
    q = 10
    d = 1
    real1 = DNAFourierRealisation(q, d)
    rng = np.random.default_rng(126)
    
    real1.coefficient = rng.standard_normal(real1.dimension)
    _ = real1.evaluate_native()
    buf1_id = id(real1._nativeBuffer)
    
    real2 = real1.clone()
    # At this point real2._nativeBuffer should be None (lazy allocation)
    assert real2._nativeBuffer is None
    
    _ = real2.evaluate_native()
    buf2_id = id(real2._nativeBuffer)
    
    assert buf1_id != buf2_id
    
    # Modify real2 and check real1 is unchanged
    old_res1 = real1.evaluate_native().copy()
    real2.coefficient = rng.standard_normal(real2.dimension)
    _ = real2.evaluate_native()
    
    assert np.all(real1.evaluate_native() == old_res1)
    print("Clone buffer isolation PASSED")

def test_buffer_batch_resize():
    """Verify that buffers correctly resize when switching between single and batch."""
    q = 10
    d = 1
    real = DNAFourierRealisation(q, d)
    rng = np.random.default_rng(127)
    
    # Single
    real.coefficient = rng.standard_normal(real.dimension)
    res_single = real.evaluate_native()
    assert res_single.ndim == 1
    
    # Batch
    nBatch = 3
    thetas = rng.standard_normal((nBatch, real.dimension))
    real.coefficient = thetas
    res_batch = real.evaluate_native()
    assert res_batch.shape == (nBatch, q + 2)
    
    # Back to single
    real.coefficient = rng.standard_normal(real.dimension)
    res_single_back = real.evaluate_native()
    assert res_single_back.ndim == 1
    
    print("Buffer batch resize PASSED")

if __name__ == "__main__":
    test_component_buffer_reuse()
    test_realisation_buffer_reuse()
    test_buffer_numerical_consistency()
    test_clone_isolation()
    test_buffer_batch_resize()
    print("\nALL BUFFER TESTS PASSED")
