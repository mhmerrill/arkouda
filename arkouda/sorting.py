from __future__ import annotations
from typing import cast, Sequence, Union
from typeguard import typechecked, check_type
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.pdarraycreation import zeros
from arkouda.strings import Strings
from arkouda.dtypes import int64, float64

numeric_dtypes = {float64,int64}

__all__ = ["argsort", "coargsort", "sort"]

def argsort(pda : Union[pdarray,Strings,'Categorical']) -> pdarray: # type: ignore
    """
    Return the permutation that sorts the array.
    
    Parameters
    ----------
    pda : pdarray or Strings or Categorical
        The array to sort (int64 or float64)

    Returns
    -------
    pdarray, int64
        The indices such that ``pda[indices]`` is sorted
        
    Raises
    ------
    TypeError
        Raised if the parameter is other than a pdarray or Strings

    See Also
    --------
    coargsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and
    resilient to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10)
    >>> perm = ak.argsort(a)
    >>> a[perm]
    array([0, 1, 1, 3, 4, 5, 7, 8, 8, 9])
    """
    from arkouda.categorical import Categorical
    check_type(argname='argsort', value=pda, 
                      expected_type=Union[pdarray,Strings,Categorical])
    if hasattr(pda, "argsort"):
        return cast(Categorical,pda).argsort()
    if pda.size == 0:
        return zeros(0, dtype=int64)
    if isinstance(pda, Strings):
        name = '{}+{}'.format(pda.entry.name, "legacy_placeholder")
    else:
        name = pda.name
    repMsg = generic_msg(cmd="argsort", args="{} {}".format(pda.objtype, name))
    return create_pdarray(cast(str,repMsg))


def coargsort(arrays: Sequence[Union[Strings, pdarray, 'Categorical']]) -> pdarray:  # type: ignore
    """
    Return the permutation that groups the rows (left-to-right), if the
    input arrays are treated as columns. The permutation sorts numeric
    columns, but not strings/Categoricals -- strings/Categoricals are grouped, but not ordered.
    
    Parameters
    ----------
    arrays : Sequence[Union[Strings, pdarray, Categorical]]
        The columns (int64, float64, Strings, or Categorical) to sort by row

    Returns
    -------
    pdarray, int64
        The indices that permute the rows to grouped order
        
    Raises
    ------
    ValueError
        Raised if the pdarrays are not of the same size or if the parameter
        is not an Iterable containing pdarrays, Strings, or Categoricals

    See Also
    --------
    argsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive. Starts with the
    last array and moves forward. This sort operates directly on numeric types,
    but for Strings, it operates on a hash. Thus, while grouping of equivalent
    strings is guaranteed, lexicographic ordering of the groups is not. For Categoricals,
    coargsort sorts based on Categorical.codes which guarantees grouping of equivalent categories
    but not lexicographic ordering of those groups.

    Examples
    --------
    >>> a = ak.array([0, 1, 0, 1])
    >>> b = ak.array([1, 1, 0, 0])
    >>> perm = ak.coargsort([a, b])
    >>> perm
    array([2, 0, 3, 1])
    >>> a[perm]
    array([0, 0, 1, 1])
    >>> b[perm]
    array([0, 1, 0, 1])
    """
    from arkouda.categorical import Categorical
    check_type(argname='coargsort', value=arrays, expected_type=Sequence[Union[pdarray, Strings, Categorical]])
    size = -1
    anames = []
    atypes = []
    for a in arrays:
        if isinstance(a, pdarray):
            anames.append('+'.join(a._list_component_names()))
            atypes.append(a.objtype)
        elif isinstance(a, Categorical):
            anames.append(a.codes.name)
            atypes.append(a.objtype)
        elif isinstance(a, Strings):
            atypes.append(a.objtype)
            anames.append('{}+{}'.format(a.entry.name, "legacy_placeholder"))
        else:
            raise ValueError("Argument must be an iterable of pdarrays, Strings, or Categoricals")
        if size == -1:
            size = a.size
        elif size != a.size:
            raise ValueError("All pdarrays, Strings, or Categoricals must be of the same size")
    if size == 0:
        return zeros(0, dtype=int64)
    repMsg = generic_msg(cmd="coargsort", args="{:n} {} {}".format(len(arrays), 
                                                                   ' '.join(anames), ' '.join(atypes)))
    return create_pdarray(cast(str, repMsg))

@typechecked
def sort(pda : pdarray) -> pdarray:
    """
    Return a sorted copy of the array. Only sorts numeric arrays; 
    for Strings, use argsort.
    
    Parameters
    ----------
    pda : pdarray or Categorical
        The array to sort (int64 or float64)

    Returns
    -------
    pdarray, int64 or float64
        The sorted copy of pda

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    ValueError
        Raised if sort attempted on a pdarray with an unsupported dtype
        such as bool

    See Also
    --------
    argsort

    Notes
    -----
    Uses a least-significant-digit radix sort, which is stable and resilient
    to non-uniformity in data but communication intensive.

    Examples
    --------
    >>> a = ak.randint(0, 10, 10)
    >>> sorted = ak.sort(a)
    >>> a
    array([0, 1, 1, 3, 4, 5, 7, 8, 8, 9])
    """
    if pda.size == 0:
        return zeros(0, dtype=int64)
    if pda.dtype not in numeric_dtypes:
        raise ValueError("ak.sort supports float64 or int64, not {}".format(pda.dtype))
    repMsg = generic_msg(cmd="sort", args="{}".format(pda.name))
    return create_pdarray(cast(str,repMsg))
