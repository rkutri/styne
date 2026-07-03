import logging
from typing import Protocol, Optional, runtime_checkable
from tqdm import tqdm

logger = logging.getLogger(__name__)


@runtime_checkable
class ProgressReporter(Protocol):
    """
    Protocol for progress reporting, structural, no inheritance required.

    Notes
    -----
    Implementations provide `update`, `close`, and context-manager support
    (`__enter__`/`__exit__`). See flag 2 above on why the concrete classes
    in this file don't explicitly inherit this Protocol.
    """
    def update(self, stepCount: int = 1, **postfix):
        ...

    def __enter__(self):
        ...

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        ...

    def close(self):
        ...


class NullProgress:
    """
    No-op progress reporter, for when progress reporting is disabled.
    """
    def update(self, stepCount: int = 1, **postfix):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        pass

    def close(self):
        pass


class TqdmProgress:
    """
    tqdm-backed progress reporter.

    Parameters
    ----------
    total : int
        Total step count for the bar.
    description : str, optional
        Bar label.
    """
    def __init__(self, total: int, description: Optional[str] = None):
        self._bar = tqdm(total=total, desc=description, disable=False)
        self._total = total

    def update(self, stepCount: int = 1, **postfix):
        if postfix:
            self._bar.set_postfix(**postfix, refresh=False)
        self._bar.update(stepCount)

    def __enter__(self):
        return self

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        self.close()

    def close(self):
        self._bar.close()
        postfixText = ""
        if self._bar.postfix:
            cleanedPostfix = str(self._bar.postfix).strip(", ")
            postfixText = f" ({cleanedPostfix})"
        descriptionText = self._bar.desc if self._bar.desc else "Progress"
        logger.info(f"{descriptionText}: run complete{postfixText}")


def make_reporter(progress, total: int, description: Optional[str] = None) -> ProgressReporter:
    """
    Construct the appropriate progress reporter.

    Parameters
    ----------
    progress : bool | ProgressReporter
        False or None gives a `NullProgress`. An existing `ProgressReporter`
        is passed through unchanged. Any other truthy value gives a new
        `TqdmProgress`.
    total : int
        Total step count, passed to `TqdmProgress` if one is constructed.
    description : str, optional
        Bar label, passed to `TqdmProgress` if one is constructed.

    Returns
    -------
    ProgressReporter
    """
    if not progress:
        return NullProgress()
    if isinstance(progress, ProgressReporter):
        return progress
    return TqdmProgress(total=total, description=description)
