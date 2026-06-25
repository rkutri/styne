import pytest
import logging
from styne.mcmc.method.mrw import MRWFactory
from styne.parameter.vector import Vector
import numpy as np
from tests.testSetup import GaussianTargetDensity
from styne.statistics.covariance import IIDCovarianceMatrix
from styne import enable_logging

def test_silent_by_default_zero_byte_emission(capsys):
    mean = Vector([0.0, 0.0])
    cov = np.eye(2)
    target = GaussianTargetDensity(mean, cov)
    
    factory = MRWFactory()
    factory.target = target
    factory.proposalCovariance = IIDCovarianceMatrix(2, 0.1)
    sampler = factory.create()
    
    initialState = Vector([1.0, 1.0])
    sampler.run(10, initialState)
    
    captured = capsys.readouterr()
    assert captured.out == "", f"Expected no stdout, got {len(captured.out)} bytes"
    assert captured.err == "", f"Expected no stderr, got {len(captured.err)} bytes"

def test_enable_logging():
    enable_logging(logging.DEBUG)
    
    styneLogger = logging.getLogger('styne')
    assert styneLogger.level == logging.DEBUG
    assert any(isinstance(h, logging.StreamHandler) for h in styneLogger.handlers)
