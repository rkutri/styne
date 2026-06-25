import pytest
import logging
from unittest.mock import patch, MagicMock
from styne.utility.progress import make_reporter, NullProgress, TqdmProgress, ProgressReporter

class MockReporter:
    def update(self, stepCount: int = 1, **postfix):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        pass

    def close(self):
        pass

def test_make_reporter_false():
    reporter = make_reporter(False, 100)
    assert isinstance(reporter, NullProgress)

def test_make_reporter_none():
    reporter = make_reporter(None, 100)
    assert isinstance(reporter, NullProgress)

def test_make_reporter_true():
    reporter = make_reporter(True, 100, "test")
    assert isinstance(reporter, TqdmProgress)

def test_make_reporter_custom():
    custom = MockReporter()
    reporter = make_reporter(custom, 100)
    assert reporter is custom
    assert isinstance(custom, ProgressReporter)

def test_null_progress():
    progress = NullProgress()
    with progress as p:
        p.update(10)
    progress.close()

@patch('styne.utility.progress.tqdm')
def test_tqdm_progress_update(mock_tqdm):
    mock_bar = MagicMock()
    mock_tqdm.return_value = mock_bar

    progress = TqdmProgress(100, "description")
    
    progress.update(5, acc=0.5)
    
    mock_bar.set_postfix.assert_called_once_with(acc=0.5, refresh=False)
    mock_bar.update.assert_called_once_with(5)

def test_tqdm_progress_close(caplog):
    caplog.set_level(logging.INFO, logger='styne')
    
    progress = TqdmProgress(100, "desc")
    # Setting postfix internal state as tqdm would usually do it
    progress._bar.postfix = "acc=0.99"
    
    progress.close()
    
    assert any("run complete" in record.message for record in caplog.records)
    assert any("acc=0.99" in record.message for record in caplog.records)
