"""
Common import checks and matplotlib/joblib boilerplate for manuscript scripts.
"""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hasMatplotlib = True
except ImportError:
    plt = None
    hasMatplotlib = False

try:
    import joblib
    hasJoblib = True
except ImportError:
    joblib = None
    hasJoblib = False
