class Chain:
    """Backend-native trajectory and annotation storage for one MCMC chain.

    Numerical samples are retained exactly as supplied. Conversion for
    plotting or reporting belongs to the caller, outside MCMC execution.
    """

    def __init__(self):
        self._trajectory = []
        self._annotations = None

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def length(self):
        return len(self._trajectory)

    @property
    def annotations(self):
        return self._annotations

    def enable_annotations(self):
        """Enable per-step annotation tracking.

        Once enabled, 'append' stores an annotation alongside every trajectory
        entry, maintaining a strict 1-to-1 correspondence. Has no effect if
        annotation tracking is already enabled.
        """
        if self._annotations is None:
            self._annotations = []

    def append(self, stateVector, annotation=None):
        """Append a state to the trajectory, with an optional annotation.

        Parameters
        ----------
        stateVector : array
            Backend-native coordinate vector to append without conversion.
        annotation : object, optional
            Attached only if `enable_annotations` has been called.
        """
        self._trajectory.append(stateVector)
        if self._annotations is not None:
            self._annotations.append(annotation)

    def clear(self):
        self._trajectory = []
        if self._annotations is not None:
            self._annotations = []


class GibbsChain:
    """Aggregate of per-block Chain objects for Gibbs samplers.

    All blocks are appended atomically once per sweep, so they are always
    equal-length by construction.

    Parameters
    ----------
    nBlocks : int
        Number of blocks in the compound state.
    """

    def __init__(self, nBlocks: int):
        self._blocks = [Chain() for _ in range(nBlocks)]

    @property
    def nBlocks(self) -> int:
        return len(self._blocks)

    @property
    def length(self) -> int:
        return self._blocks[0].length

    def block(self, idx: int) -> Chain:
        return self._blocks[idx]

    def append(self, blockCoordinates: list):
        """Append one full compound state atomically."""
        for i, vec in enumerate(blockCoordinates):
            self._blocks[i].append(vec)

    def clear(self):
        for b in self._blocks:
            b.clear()
