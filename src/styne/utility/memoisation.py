from abc import ABC, abstractmethod
from styne.parameter.parameter import Parameter


class Cache(ABC):
    """
    Template class for caching mechanisms.

    Parameters
    ----------
    cacheSize : int
        Maximum number of elements the cache can hold.
    """

    def __init__(self, cacheSize: int) -> None:
        """
        Initialize the cache with a maximum size.

        Parameters:
            cacheSize (int): Maximum number of elements the cache can hold.
        """
        self._maxSize = cacheSize

        self._misses = 0
        self._hits = 0

        self._keys = []

    @property
    def misses(self):
        """
        Number of `retrieve` calls that missed.
        """
        return self._misses

    @property
    def hits(self):
        """
        Number of `retrieve` calls that hit.
        """
        return self._hits

    def contains(self, parameter: Parameter) -> bool:
        """
        Whether `parameter` is currently cached.

        Base implementation checks membership in `self._keys`, which the only
        current subclass, `EvaluationCache`, doesn't populate, it overrides this
        method with its own `id()`-based check instead. A new subclass relying
        on the base implementation needs to populate `_keys` itself, it isn't
        done automatically by `__init__`.

        Parameters
        ----------
        parameter : Parameter

        Returns
        -------
        bool
        """
        return parameter in self._keys

    @abstractmethod
    def add(self, parameter: Parameter, cacheValue) -> None:
        pass

    @abstractmethod
    def retrieve(self, parameter: Parameter):
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class EvaluationCache(Cache):
    """
    Fixed-size evaluation cache, keyed by `id(parameter)` rather than value
    equality. Evicts the oldest entry (FIFO, not LRU) once `cacheSize` is
    reached. Does not use the base `Cache`'s `_keys` mechanism.

    Parameters
    ----------
    cacheSize : int
        Maximum number of cached entries.
    """

    def __init__(self, cacheSize: int) -> None:
        super().__init__(cacheSize)
        self._cache = {}
        self._order = []
        self._refs = {}

    def _evict_oldest(self):
        oldId = self._order.pop(0)
        del self._cache[oldId]
        del self._refs[oldId]

    def contains(self, parameter: Parameter) -> bool:
        return id(parameter) in self._cache

    def add(self, parameter: Parameter, cacheValue) -> None:
        paramId = id(parameter)
        if paramId in self._cache:
            return
        if len(self._cache) >= self._maxSize:
            self._evict_oldest()
        self._cache[paramId] = cacheValue
        self._order.append(paramId)
        self._refs[paramId] = parameter

    def retrieve(self, parameter: Parameter):
        paramId = id(parameter)
        if paramId in self._cache:
            self._hits += 1
            return self._cache[paramId]
        self._misses += 1
        raise RuntimeError("Evaluation Cache missed.")

    def clear(self) -> None:
        self._cache = {}
        self._order = []
        self._refs = {}


# class AEMCache(Cache):
#     """
#     Cache specifically designed to store AEM-relevant evaluations.
#     """
#
#     # Large cache can become memory-costly quickly, as we store evaluations of
#     # the forward model.
#     CACHESIZE = 3
#
#     def __init__(self) -> None:
#
#         super().__init__(AEMCache.CACHESIZE)
#
#         self._fmCache = []  # Cache for forward model evaluations
#         self._llCache = []  # Cache for log-likelihood evaluations
#
#     def _evict_oldest(self):
#
#         self._keys.pop(0)
#         self._fmCache.pop(0)
#         self._llCache.pop(0)
#
#     def _move_to_back(self, index):
#         """Move the cached element at 'index' to the back of all caches."""
#
#         self._keys.append(self._keys.pop(index))
#         self._fmCache.append(self._fmCache.pop(index))
#         self._llCache.append(self._llCache.pop(index))
#
#     def add(self, parameter: Parameter,
#             cacheValue: AEMEvaluation) -> None:
#
#         if not isinstance(cacheValue, AEMEvaluation):
#             raise ValueError("AEM Cache is supposed to store AEM-relevant "
#                              "evaluations.")
#
#         if len(self._keys) >= self._maxSize:
#             self._evict_oldest()
#
#         self._keys.append(parameter)
#         self._fmCache.append(cacheValue.forwardModelEvaluation)
#         self._llCache.append(cacheValue.logLikelihoodEvaluation)
#
#     def retrieve(self, parameter: Parameter):
#         """
#         Retrieve the forward model and log-likelihood evaluations for a
#         parameter.
#
#         Parameters:
#             parameter (Parameter): The parameter whose evaluations to
#                                             retrieve.
#
#         Returns:
#             tuple: A tuple containing
#                    (forwardModelEvaluation, logLikelihoodEvaluation).
#
#         Raises:
#             RuntimeError: If the parameter is not in the cache.
#         """
#         if self.contains(parameter):
#             self._hits += 1
#             paramIdx = self._keys.index(parameter)
#
#             # Move the retrieved item to the back of the cache
#             self._move_to_back(paramIdx)
#
#             fmEval = self._fmCache[-1]
#             llEval = self._llCache[-1]
#
#             return fmEval, llEval
#
#         self._misses += 1
#         raise RuntimeError("Retrieving evaluation for parameter failed "
#                            "(cache miss).")
