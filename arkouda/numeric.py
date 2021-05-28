import numpy as np # type: ignore
from typeguard import typechecked
from typing import cast as type_cast
from typing import Optional, Tuple, Union, ForwardRef
from arkouda.client import generic_msg
from arkouda.dtypes import resolve_scalar_dtype, DTypes, isSupportedNumber, \
     int_scalars, numeric_scalars
from arkouda.dtypes import _as_dtype
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.pdarraysetops import unique
from arkouda.strings import Strings

Categorical = ForwardRef('Categorical')

__all__ = ["cast", "abs", "log", "exp", "cumsum", "cumprod", "sin", "cos", 
           "where", "histogram", "value_counts", "isnan"]

@typechecked
def cast(pda : Union[pdarray, Strings], dt: Union[np.dtype,str]) -> Union[pdarray, Strings]:
    """
    Cast an array to another dtype.

    Parameters
    ----------
    pda : pdarray or Strings
        The array of values to cast
    dtype : np.dtype or str
        The target dtype to cast values to

    Returns
    -------
    pdarray or Strings
        Array of values cast to desired dtype

    Notes
    -----
    The cast is performed according to Chapel's casting rules and is NOT safe 
    from overflows or underflows. The user must ensure that the target dtype 
    has the precision and capacity to hold the desired result.
    
    Examples
    --------
    >>> ak.cast(ak.linspace(1.0,5.0,5), dt=ak.int64)
    array([1, 2, 3, 4, 5])    
    
    >>> ak.cast(ak.arange(0,5), dt=ak.float64).dtype
    dtype('float64')
    
    >>> ak.cast(ak.arange(0,5), dt=ak.bool)
    array([False, True, True, True, True])
    
    >>> ak.cast(ak.linspace(0,4,5), dt=ak.bool)
    array([False, True, True, True, True])
    """

    if isinstance(pda, pdarray):
        name = pda.name
        objtype = "pdarray"
    elif isinstance(pda, Strings):
        name = pda.entry.name
        objtype = "str"    
    # typechecked decorator guarantees no other case

    dt = _as_dtype(dt)
    opt = ""
    cmd = "cast"
    args= "{} {} {} {}".format(name, objtype, dt.name, opt)
    repMsg = generic_msg(cmd=cmd,args=args)
    if dt.name.startswith("str"):
        return Strings.from_parts(*(type_cast(str,repMsg).split("+")))
    else:
        return create_pdarray(type_cast(str,repMsg))

@typechecked
def abs(pda : pdarray) -> pdarray:
    """
    Return the element-wise absolute value of the array.

    Parameters
    ----------
    pda : pdarray
    
    Returns
    -------
    pdarray
        A pdarray containing absolute values of the input array elements
   
    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
        
    Examples
    --------
    >>> ak.abs(ak.arange(-5,-1))
    array([5, 4, 3, 2])
    
    >>> ak.abs(ak.linspace(-5,-1,5))
    array([5, 4, 3, 2, 1])    
    """
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("abs", pda.name))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def log(pda : pdarray) -> pdarray:
    """
    Return the element-wise natural log of the array. 

    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing natural log values of the input 
        array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    Notes
    -----
    Logarithms with other bases can be computed as follows:

    Examples
    --------
    >>> A = ak.array([1, 10, 100])
    # Natural log
    >>> ak.log(A)
    array([0, 2.3025850929940459, 4.6051701859880918])
    # Log base 10
    >>> ak.log(A) / np.log(10)
    array([0, 1, 2])
    # Log base 2
    >>> ak.log(A) / np.log(2)
    array([0, 3.3219280948873626, 6.6438561897747253])
    """
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("log", pda.name))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def exp(pda : pdarray) -> pdarray:
    """
    Return the element-wise exponential of the array.
    
    Parameters
    ----------
    pda : pdarray

    Returns
    -------
    pdarray
        A pdarray containing exponential values of the input 
        array elements

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
        
    Examples
    --------
    >>> ak.exp(ak.arange(1,5))
    array([2.7182818284590451, 7.3890560989306504, 20.085536923187668, 54.598150033144236])
    
    >>> ak.exp(ak.uniform(5,1.0,5.0))
    array([11.84010843172504, 46.454368507659211, 5.5571769623557188, 
           33.494295836924771, 13.478894913238722])
    """
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("exp", pda.name))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def cumsum(pda : pdarray) -> pdarray:
    """
    Return the cumulative sum over the array. 

    The sum is inclusive, such that the ``i`` th element of the 
    result is the sum of elements up to and including ``i``.
    
    Parameters
    ----------
    pda : pdarray
    
    Returns
    -------
    pdarray
        A pdarray containing cumulative sums for each element
        of the original pdarray
    
    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
        
    Examples
    --------
    >>> ak.cumsum(ak.arange([1,5]))
    array([1, 3, 6])

    >>> ak.cumsum(ak.uniform(5,1.0,5.0))
    array([3.1598310770203937, 5.4110385860243131, 9.1622479306453748, 
           12.710615785506533, 13.945880905466208])
    
    >>> ak.cumsum(ak.randint(0, 1, 5, dtype=ak.bool))
    array([0, 1, 1, 2, 3])
    """
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("cumsum", pda.name))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def cumprod(pda : pdarray) -> pdarray:
    """
    Return the cumulative product over the array. 

    The product is inclusive, such that the ``i`` th element of the 
    result is the product of elements up to and including ``i``.

    Parameters
    ----------
    pda : pdarray
    
    Returns
    -------
    pdarray
        A pdarray containing cumulative products for each element
        of the original pdarray

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
        
    Examples
    --------
    >>> ak.cumprod(ak.arange(1,5))
    array([1, 2, 6, 24]))

    >>> ak.cumprod(ak.uniform(5,1.0,5.0))
    array([1.5728783400481925, 7.0472855509390593, 33.78523998586553, 
           134.05309592737584, 450.21589865655358])
    """
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("cumprod", pda.name))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def sin(pda : pdarray) -> pdarray:
    """
    Return the element-wise sine of the array.

    Parameters
    ----------
    pda : pdarray
    
    Returns
    -------
    pdarray
        A pdarray containing sin for each element
        of the original pdarray
    
    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    repMsg = generic_msg(cmd="efunc", args="{} {}".format("sin",pda.name))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def cos(pda : pdarray) -> pdarray:
    """
    Return the element-wise cosine of the array.

    Parameters
    ----------
    pda : pdarray
    
    Returns
    -------
    pdarray
        A pdarray containing cosine for each element
        of the original pdarray
    
    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    """
    repMsg = type_cast(str, generic_msg(cmd="efunc", args="{} {}".format("cos",pda.name)))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def where(condition : pdarray, A : Union[numeric_scalars, pdarray], 
                        B : Union[numeric_scalars, pdarray]) -> pdarray:
    """
    Returns an array with elements chosen from A and B based upon a 
    conditioning array. As is the case with numpy.where, the return array
    consists of values from the first array (A) where the conditioning array 
    elements are True and from the second array (B) where the conditioning
    array elements are False.
    
    Parameters
    ----------
    condition : pdarray
        Used to choose values from A or B
    A : Union[numeric_scalars, pdarray]
        Value(s) used when condition is True
    B : Union[numeric_scalars, pdarray]
        Value(s) used when condition is False

    Returns
    -------
    pdarray
        Values chosen from A where the condition is True and B where
        the condition is False
        
    Raises 
    ------
    TypeError
        Raised if the condition object is not a pdarray, if A or B is not
        an int, np.int64, float, np.float64, or pdarray, if pdarray dtypes 
        are not supported or do not match, or multiple condition clauses (see 
        Notes section) are applied
    ValueError
        Raised if the shapes of the condition, A, and B pdarrays are unequal
        
    Examples
    --------
    >>> a1 = ak.arange(1,10)
    >>> a2 = ak.ones(9, dtype=np.int64)
    >>> cond = a1 < 5
    >>> ak.where(cond,a1,a2)
    array([1, 2, 3, 4, 1, 1, 1, 1, 1])
    
    >>> a1 = ak.arange(1,10)
    >>> a2 = ak.ones(9, dtype=np.int64)
    >>> cond = a1 == 5
    >>> ak.where(cond,a1,a2)
    array([1, 1, 1, 1, 5, 1, 1, 1, 1])

    >>> a1 = ak.arange(1,10)
    >>> a2 = 10
    >>> cond = a1 < 5
    >>> ak.where(cond,a1,a2)
    array([1, 2, 3, 4, 10, 10, 10, 10, 10])

    Notes
    -----
    A and B must have the same dtype and only one conditional clause 
    is supported e.g., n < 5, n > 1, which is supported in numpy
    is not currently supported in Arkouda
    """
    if (not isSupportedNumber(A) and not isinstance(A,pdarray)) or \
                                      (not isSupportedNumber(B) and not isinstance(B,pdarray)):
        raise TypeError('both A and B must be an int, np.int64, float, np.float64, or pdarray')
    if isinstance(A, pdarray) and isinstance(B, pdarray):
        repMsg = generic_msg(cmd="efunc3vv", args="{} {} {} {}".\
                             format("where",
                                    condition.name,
                                    A.name,
                                    B.name))
    # For scalars, try to convert it to the array's dtype
    elif isinstance(A, pdarray) and np.isscalar(B):
        repMsg = generic_msg(cmd="efunc3vs", args="{} {} {} {} {}".\
                             format("where",
                                    condition.name,
                                    A.name,
                                    A.dtype.name,
                                    A.format_other(B)))
    elif isinstance(B, pdarray) and np.isscalar(A):
        repMsg = generic_msg(cmd="efunc3sv", args="{} {} {} {} {}".\
                             format("where",
                                    condition.name,
                                    B.dtype.name,
                                    B.format_other(A),
                                    B.name))
    elif np.isscalar(A) and np.isscalar(B):
        # Scalars must share a common dtype (or be cast)
        dtA = resolve_scalar_dtype(A)
        dtB = resolve_scalar_dtype(B)
        # Make sure at least one of the dtypes is supported
        if not (dtA in DTypes or dtB in DTypes):
            raise TypeError(("Not implemented for scalar types {} " +
                            "and {}").format(dtA, dtB))
        # If the dtypes are the same, do not cast
        if dtA == dtB: # type: ignore
            dt = dtA
        # If the dtypes are different, try casting one direction then the other
        elif dtB in DTypes and np.can_cast(A, dtB):
            A = np.dtype(dtB).type(A)
            dt = dtB
        elif dtA in DTypes and np.can_cast(B, dtA):
            B = np.dtype(dtA).type(B)
            dt = dtA
        # Cannot safely cast
        else:
            raise TypeError(("Cannot cast between scalars {} and {} to " +
                            "supported dtype").format(A, B))
        repMsg = generic_msg(cmd="efunc3ss", args="{} {} {} {} {} {}".\
                             format("where",
                                    condition.name,
                                    dt,
                                    A,
                                    dt,
                                    B))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def histogram(pda : pdarray, bins : int_scalars=10) -> pdarray:
    """
    Compute a histogram of evenly spaced bins over the range of an array.
    
    Parameters
    ----------
    pda : pdarray
        The values to histogram

    bins : int_scalars
        The number of equal-size bins to use (default: 10)

    Returns
    -------
    pdarray, int64 or float64
        The number of values present in each bin
        
    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray or if bins is
        not an int.
    ValueError
        Raised if bins < 1
    NotImplementedError
        Raised if pdarray dtype is bool or uint8

    See Also
    --------
    value_counts

    Notes
    -----
    The bins are evenly spaced in the interval [pda.min(), pda.max()].
    Currently, the user must re-compute the bin edges, e.g. with np.linspace 
    (see below) in order to plot the histogram.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> A = ak.arange(0, 10, 1)
    >>> nbins = 3
    >>> h = ak.histogram(A, bins=nbins)
    >>> h
    array([3, 3, 4])
    # Recreate the bin edges in NumPy
    >>> binEdges = np.linspace(A.min(), A.max(), nbins+1)
    >>> binEdges
    array([0., 3., 6., 9.])
    # To plot, use only the left edges, and export the histogram to NumPy
    >>> plt.plot(binEdges[:-1], h.to_ndarray())
    """
    if bins < 1:
        raise ValueError('bins must be 1 or greater')
    repMsg = generic_msg(cmd="histogram", args="{} {}".format(pda.name, bins))
    return create_pdarray(type_cast(str,repMsg))

@typechecked
def value_counts(pda : pdarray) -> Union[Categorical, # type: ignore
                        Tuple[Union[pdarray,Strings],Optional[pdarray]]]:
    """
    Count the occurrences of the unique values of an array.

    Parameters
    ----------
    pda : pdarray, int64
        The array of values to count

    Returns
    -------
    unique_values : pdarray, int64 or Strings
        The unique values, sorted in ascending order

    counts : pdarray, int64
        The number of times the corresponding unique value occurs
        
    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray

    See Also
    --------
    unique, histogram

    Notes
    -----
    This function differs from ``histogram()`` in that it only returns
    counts for values that are present, leaving out empty "bins". This
    function delegates all logic to the unique() method where the 
    return_counts parameter is set to True.

    Examples
    --------
    >>> A = ak.array([2, 0, 2, 4, 0, 0])
    >>> ak.value_counts(A)
    (array([0, 2, 4]), array([3, 2, 1]))
    """
    return unique(pda, return_counts=True)


@typechecked
def isnan(pda : pdarray) -> pdarray:
    """
    Test a pdarray for Not a number / NaN values
    Currently only supports float-value-based arrays

    Parameters
    ----------
    pda : pdarray to test

    Returns
    -------
    pdarray consisting of True / False values; True where NaN, False otherwise

    Raises
    ------
    TypeError
        Raised if the parameter is not a pdarray
    RuntimeError
        if the underlying pdarray is not float-based
    """
    rep_msg = generic_msg(cmd="efunc", args=f"isnan {pda.name}")
    return create_pdarray(type_cast(str, rep_msg))
