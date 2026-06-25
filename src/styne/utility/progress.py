import logging
from typing import Protocol, Optional, runtime_checkable
from tqdm import tqdm

logger = logging.getLogger(__name__)


@runtime_checkable
class ProgressReporter(Protocol):
    def update(self, stepCount: int = 1, **postfix):
        ...

    def __enter__(self):
        ...

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        ...

    def close(self):
        ...


class NullProgress:
    def update(self, stepCount: int = 1, **postfix):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exceptionType, exceptionValue, exceptionTraceback):
        pass

    def close(self):
        pass


class TqdmProgress:
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
    if not progress:
        return NullProgress()
    if isinstance(progress, ProgressReporter):
        return progress
    return TqdmProgress(total=total, description=description)
