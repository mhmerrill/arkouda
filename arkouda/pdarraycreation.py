import numpy as np # type: ignore
import pandas as pd # type: ignore
from typing import cast, Iterable, Optional, Union
from typeguard import typechecked
from arkouda.client import generic_msg
from arkouda.dtypes import structDtypeCodes, NUMBER_FORMAT_STRINGS, float64, int64, \
     DTypes, isSupportedInt, isSupportedNumber, NumericDTypes, SeriesDTypes,\
    int_scalars, numeric_scalars, get_byteorder
from arkouda.dtypes import dtype as akdtype
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.strings import Strings

__all__ = ["array", "zeros", "ones", "zeros_like", "ones_like", 
           "arange", "linspace", "randint", "uniform", "standard_normal",
           "random_strings_uniform", "random_strings_lognormal", 
           "from_series"
          ]

@typechecked
def from_series(series : pd.Series, 
                    dtype : Optional[Union[type,str]]=None) -> Union[pdarray,Strings]:
    """
    Converts a Pandas Series to an Arkouda pdarray or Strings object. If
    dtype is None, the dtype is inferred from the Pandas Series. Otherwise,
    the dtype parameter is set if the dtype of the Pandas Series is to be 
    overridden or is  unknown (for example, in situations where the Series 
    dtype is object).
    
    Parameters
    ----------
    series : Pandas Series
        The Pandas Series with a dtype of bool, float64, int64, or string
    dtype : Optional[type]
        The valid dtype types are np.bool, np.float64, np.int64, and np.str

    Returns
    -------
    Union[pdarray,Strings]
    
    Raises
    ------
    TypeError
        Raised if series is not a Pandas Series object
    ValueError
        Raised if the Series dtype is not bool, float64, int64, string, datetime, or timedelta

    Examples
    --------
    >>> ak.from_series(pd.Series(np.random.randint(0,10,5)))
    array([9, 0, 4, 7, 9])

    >>> ak.from_series(pd.Series(['1', '2', '3', '4', '5']),dtype=np.int64)
    array([1, 2, 3, 4, 5])

    >>> ak.from_series(pd.Series(np.random.uniform(low=0.0,high=1.0,size=3)))
    array([0.57600036956445599, 0.41619265571741659, 0.6615356693784662])

    >>> ak.from_series(pd.Series(['0.57600036956445599', '0.41619265571741659',
                       '0.6615356693784662']), dtype=np.float64)
    array([0.57600036956445599, 0.41619265571741659, 0.6615356693784662])

    >>> ak.from_series(pd.Series(np.random.choice([True, False],size=5)))
    array([True, False, True, True, True])

    >>> ak.from_series(pd.Series(['True', 'False', 'False', 'True', 'True']), dtype=np.bool)
    array([True, True, True, True, True])

    >>> ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e'], dtype="string"))
    array(['a', 'b', 'c', 'd', 'e'])

    >>> ak.from_series(pd.Series(['a', 'b', 'c', 'd', 'e']),dtype=np.str)
    array(['a', 'b', 'c', 'd', 'e'])

    >>> ak.from_series(pd.Series(pd.to_datetime(['1/1/2018', np.datetime64('2018-01-01')])))
    array([1514764800000000000, 1514764800000000000])  
    
    Notes
    -----
    The supported datatypes are bool, float64, int64, string, and datetime64[ns]. The
    data type is either inferred from the the Series or is set via the dtype parameter. 
    
    Series of datetime or timedelta are converted to Arkouda arrays of dtype int64 (nanoseconds)
    
    A Pandas Series containing strings has a dtype of object. Arkouda assumes the Series
    contains strings and sets the dtype to str
    """ 
    if not dtype:   
        dt = series.dtype.name
    else:
        dt = str(dtype)
    try:
        '''
        If the Series has a object dtype, set dtype to string to comply with method
        signature that does not require a dtype; this is required because Pandas can infer
        non-str dtypes from the input np or Python array.
        '''
        if dt == 'object':
            dt = 'string'

        n_array = series.to_numpy(dtype=SeriesDTypes[dt])
    except KeyError:
        raise ValueError(('dtype {} is unsupported. Supported dtypes are bool, ' +
                      'float64, int64, string, datetime64[ns], and timedelta64[ns]').format(dt))
    return array(n_array)

def array(a : Union[pdarray,np.ndarray, Iterable]) -> Union[pdarray, Strings]:
    """
    Convert a Python or Numpy Iterable to a pdarray or Strings object, sending 
    the corresponding data to the arkouda server. 

    Parameters
    ----------
    a : Union[pdarray, np.ndarray]
        Rank-1 array of a supported dtype

    Returns
    -------
    pdarray or Strings
        A pdarray instance stored on arkouda server or Strings instance, which
        is composed of two pdarrays stored on arkouda server
        
    Raises
    ------
    TypeError
        Raised if a is not a pdarray, np.ndarray, or Python Iterable such as a
        list, array, tuple, or deque
    RuntimeError
        Raised if a is not one-dimensional, nbytes > maxTransferBytes, a.dtype is
        not supported (not in DTypes), or if the product of a size and
        a.itemsize > maxTransferBytes
    ValueError
        Raised if the returned message is malformed or does not contain the fields
        required to generate the array.

    See Also
    --------
    pdarray.to_ndarray

    Notes
    -----
    The number of bytes in the input array cannot exceed `arkouda.maxTransferBytes`,
    otherwise a RuntimeError will be raised. This is to protect the user
    from overwhelming the connection between the Python client and the arkouda
    server, under the assumption that it is a low-bandwidth connection. The user
    may override this limit by setting ak.maxTransferBytes to a larger value, 
    but should proceed with caution.
    
    If the pdrray or ndarray is of type U, this method is called twice recursively 
    to create the Strings object and the two corresponding pdarrays for string 
    bytes and offsets, respectively.

    Examples
    --------
    >>> ak.array(np.arange(1,10))
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    >>> ak.array(range(1,10))
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
   
    >>> strings = ak.array(['string {}'.format(i) for i in range(0,5)])
    >>> type(strings)
    <class 'arkouda.strings.Strings'>  
    """
    # If a is already a pdarray, do nothing
    if isinstance(a, pdarray):
        return a
    from arkouda.client import maxTransferBytes
    # If a is not already a numpy.ndarray, convert it
    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except:
            raise TypeError(('a must be a pdarray, np.ndarray, or convertible to' +
                            ' a numpy array'))
    # Only rank 1 arrays currently supported
    if a.ndim != 1:
        raise RuntimeError("Only rank-1 pdarrays or ndarrays supported")
    # Check if array of strings
    if a.dtype.kind == 'U' or  'U' in a.dtype.kind:
        encoded = np.array([elem.encode() for elem in a])
        # Length of each string, plus null byte terminator
        lengths = np.array([len(elem) for elem in encoded]) + 1
        # Compute zero-up segment offsets
        offsets = np.cumsum(lengths) - lengths
        # Allocate and fill bytes array with string segments
        nbytes = offsets[-1] + lengths[-1]
        if nbytes > maxTransferBytes:
            raise RuntimeError(("Creating pdarray would require transferring {} bytes," +
                                " which exceeds allowed transfer size. Increase " +
                                "ak.maxTransferBytes to force.").format(nbytes))
        values = np.zeros(nbytes, dtype=np.uint8)
        for s, o in zip(encoded, offsets):
            for i, b in enumerate(s):
                values[o+i] = b
        # Recurse to create pdarrays for offsets and values, then return Strings object
        return Strings(cast(pdarray, array(offsets)), cast(pdarray, array(values)))
    # If not strings, then check that dtype is supported in arkouda
    if a.dtype.name not in DTypes:
        raise RuntimeError("Unhandled dtype {}".format(a.dtype))
    # Do not allow arrays that are too large
    size = a.size
    if (size * a.itemsize) > maxTransferBytes:
        raise RuntimeError(("Array exceeds allowed transfer size. Increase " +
                            "ak.maxTransferBytes to allow"))
    # Pack binary array data into a bytes object with a command header
    # including the dtype and size. Note that the server expects big-endian so
    # if we're using litle-endian swap the bytes before sending.
    if get_byteorder(a.dtype) == '<':
        abytes = a.byteswap().tobytes()
    else:
        abytes = a.tobytes()
    req_msg = "{} {:n} ".  format(a.dtype.name, size).encode() + abytes
    repMsg = generic_msg(cmd='array', args=req_msg, send_bytes=True)
    return create_pdarray(repMsg)

def zeros(size : int_scalars, dtype : type=np.float64) -> pdarray:
    """
    Create a pdarray filled with zeros.

    Parameters
    ----------
    size : int_scalars
        Size of the array (only rank-1 arrays supported)
    dtype : all_scalars
        Type of resulting array, default float64

    Returns
    -------
    pdarray
        Zeros of the requested size and dtype
        
    Raises
    ------
    TypeError
        Raised if the supplied dtype is not supported or if the size
        parameter is neither an int nor a str that is parseable to an int.

    See Also
    --------
    ones, zeros_like

    Examples
    --------
    >>> ak.zeros(5, dtype=ak.int64)
    array([0, 0, 0, 0, 0])

    >>> ak.zeros(5, dtype=ak.float64)
    array([0, 0, 0, 0, 0])

    >>> ak.zeros(5, dtype=ak.bool)
    array([False, False, False, False, False])
    """
    if not np.isscalar(size):
        raise TypeError("size must be a scalar, not {}".\
                                     format(size.__class__.__name__))
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if cast(np.dtype,dtype).name not in NumericDTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    repMsg = generic_msg(cmd="create", args="{} {}".format(
                                    cast(np.dtype,dtype).name, size))
    
    return create_pdarray(repMsg)

def ones(size : int_scalars, dtype : type=float64) -> pdarray:
    """
    Create a pdarray filled with ones.

    Parameters
    ----------
    size : int_scalars
        Size of the array (only rank-1 arrays supported)
    dtype : Union[float64, int64, bool]
        Resulting array type, default float64

    Returns
    -------
    pdarray
        Ones of the requested size and dtype
        
    Raises
    ------
    TypeError
        Raised if the supplied dtype is not supported or if the size
        parameter is neither an int nor a str that is parseable to an int.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> ak.ones(5, dtype=ak.int64)
    array([1, 1, 1, 1, 1])

    >>> ak.ones(5, dtype=ak.float64)
    array([1, 1, 1, 1, 1])

    >>> ak.ones(5, dtype=ak.bool)
    array([True, True, True, True, True])
    """
    if not np.isscalar(size):
        raise TypeError("size must be a scalar, not {}".\
                                            format(size.__class__.__name__))
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if cast(np.dtype,dtype).name not in NumericDTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    repMsg = generic_msg(cmd="create", args="{} {}".format(
                                           cast(np.dtype,dtype).name, size))
    a = create_pdarray(repMsg)
    a.fill(1)
    return a

@typechecked
def zeros_like(pda : pdarray) -> pdarray:
    """
    Create a zero-filled pdarray of the same size and dtype as an existing 
    pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.zeros(pda.size, pda.dtype)
        
    Raises
    ------
    TypeError
        Raised if the pda parameter is not a pdarray.

    See Also
    --------
    zeros, ones_like

    Examples
    --------
    >>> zeros = ak.zeros(5, dtype=ak.int64)
    >>> ak.zeros_like(zeros)
    array([0, 0, 0, 0, 0])

    >>> zeros = ak.zeros(5, dtype=ak.float64)
    >>> ak.zeros_like(zeros)
    array([0, 0, 0, 0, 0])

    >>> zeros = ak.zeros(5, dtype=ak.bool)
    >>> ak.zeros_like(zeros)
    array([False, False, False, False, False])
    """
    return zeros(pda.size, pda.dtype)

@typechecked
def ones_like(pda : pdarray) -> pdarray:
    """
    Create a one-filled pdarray of the same size and dtype as an existing 
    pdarray.

    Parameters
    ----------
    pda : pdarray
        Array to use for size and dtype

    Returns
    -------
    pdarray
        Equivalent to ak.ones(pda.size, pda.dtype)
        
    Raises
    ------
    TypeError
        Raised if the pda parameter is not a pdarray.

    See Also
    --------
    ones, zeros_like
    
    Notes
    -----
    Logic for generating the pdarray is delegated to the ak.ones method.
    Accordingly, the supported dtypes match are defined by the ak.ones method.
    
    Examples
    --------
    >>> ones = ak.ones(5, dtype=ak.int64)
     >>> ak.ones_like(ones)
    array([1, 1, 1, 1, 1])

    >>> ones = ak.ones(5, dtype=ak.float64)
    >>> ak.ones_like(ones)
    array([1, 1, 1, 1, 1])

    >>> ones = ak.ones(5, dtype=ak.bool)
    >>> ak.ones_like(ones)
    array([True, True, True, True, True])
    """
    return ones(pda.size, pda.dtype)

def arange(*args) -> pdarray:
    """
    arange([start,] stop[, stride])

    Create a pdarray of consecutive integers within the interval [start, stop).
    If only one arg is given then arg is the stop parameter. If two args are
    given, then the first arg is start and second is stop. If three args are
    given, then the first arg is start, second is stop, third is stride.

    Parameters
    ----------
    start : int_scalars, optional
        Starting value (inclusive)
    stop : int_scalars
        Stopping value (exclusive)
    stride : int_scalars, optional
        The difference between consecutive elements, the default stride is 1,
        if stride is specified then start must also be specified. 

    Returns
    -------
    pdarray, int64
        Integers from start (inclusive) to stop (exclusive) by stride
        
    Raises
    ------
    TypeError
        Raised if start, stop, or stride is not an int object
    ZeroDivisionError
        Raised if stride == 0

    See Also
    --------
    linspace, zeros, ones, randint
    
    Notes
    -----
    Negative strides result in decreasing values. Currently, only int64 
    pdarrays can be created with this method. For float64 arrays, use 
    the linspace method.

    Examples
    --------
    >>> ak.arange(0, 5, 1)
    array([0, 1, 2, 3, 4])

    >>> ak.arange(5, 0, -1)
    array([5, 4, 3, 2, 1])

    >>> ak.arange(0, 10, 2)
    array([0, 2, 4, 6, 8])
    
    >>> ak.arange(-5, -10, -1)
    array([-5, -6, -7, -8, -9])
    """
   
    #if one arg is given then arg is stop
    if len(args) == 1:
        start = 0
        stop = args[0]
        stride = 1

    #if two args are given then first arg is start and second is stop
    if len(args) == 2:
        start = args[0]
        stop = args[1]
        stride = 1

    #if three args are given then first arg is start,
    #second is stop, third is stride
    if len(args) == 3:
        start = args[0]
        stop = args[1]
        stride = args[2]

    if stride == 0:
        raise ZeroDivisionError("division by zero")

    if isSupportedInt(start) and isSupportedInt(stop) and isSupportedInt(stride):
        if stride < 0:
            stop = stop + 2
        repMsg = generic_msg(cmd='arange', args="{} {} {}".format(start, stop, stride))
        return create_pdarray(repMsg)
    else:
        raise TypeError("start,stop,stride must be type int or np.int64 {} {} {}".\
                                    format(start,stop,stride))

@typechecked
def linspace(start : numeric_scalars, 
             stop : numeric_scalars, length : int_scalars) -> pdarray:
    """
    Create a pdarray of linearly-spaced floats in a closed interval.

    Parameters
    ----------
    start : numeric_scalars
        Start of interval (inclusive)
    stop : numeric_scalars
        End of interval (inclusive)
    length : int_scalars
        Number of points

    Returns
    -------
    pdarray, float64
        Array of evenly spaced float values along the interval
        
    Raises
    ------
    TypeError
        Raised if start or stop is not a float or int or if length is not an int

    See Also
    --------
    arange
    
    Notes
    -----
    If that start is greater than stop, the pdarray values are generated
    in descending order.

    Examples
    --------
    >>> ak.linspace(0, 1, 5)
    array([0, 0.25, 0.5, 0.75, 1])

    >>> ak.linspace(start=1, stop=0, length=5)
    array([1, 0.75, 0.5, 0.25, 0])

    >>> ak.linspace(start=-5, stop=0, length=5)
    array([-5, -3.75, -2.5, -1.25, 0])
    """
    if not isSupportedNumber(start) or not isSupportedNumber(stop):
        raise TypeError('both start and stop must be an int, np.int64, float, or np.float64')
    if not isSupportedNumber(length):
        raise TypeError('length must be an int or int64')
    repMsg = generic_msg(cmd='linspace', args="{} {} {}".format(start, stop, length))
    return create_pdarray(repMsg)

@typechecked
def randint(low : numeric_scalars, high : numeric_scalars, 
            size : int_scalars, dtype=int64, seed : int_scalars=None) -> pdarray:
    """
    Generate a pdarray of randomized int, float, or bool values in a 
    specified range bounded by the low and high parameters.

    Parameters
    ----------
    low : numeric_scalars
        The low value (inclusive) of the range
    high : numeric_scalars
        The high value (exclusive for int, inclusive for float) of the range
    size : int_scalars
        The length of the returned array
    dtype : Union[int64, float64, bool]
        The dtype of the array
    seed : int_scalars
        Index for where to pull the first returned value
        

    Returns
    -------
    pdarray
        Values drawn uniformly from the specified range having the desired dtype
        
    Raises
    ------
    TypeError
        Raised if dtype.name not in DTypes, size is not an int, low or high is
        not an int or float, or seed is not an int
    ValueError
        Raised if size < 0 or if high < low

    Notes
    -----
    Calling randint with dtype=float64 will result in uniform non-integral
    floating point values.

    Examples
    --------
    >>> ak.randint(0, 10, 5)
    array([5, 7, 4, 8, 3])

    >>> ak.randint(0, 1, 3, dtype=ak.float64)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])

    >>> ak.randint(0, 1, 5, dtype=ak.bool)
    array([True, False, True, True, True])
    
    >>> ak.randint(1, 5, 10, seed=2)
    array([4, 3, 1, 3, 4, 4, 2, 4, 3, 2])

    >>> ak.randint(1, 5, 3, dtype=ak.float64, seed=2)
    array([2.9160772326374946, 4.353429832157099, 4.5392023718621486])
    
    >>> ak.randint(1, 5, 10, dtype=ak.bool, seed=2)
    array([False, True, True, True, True, False, True, True, True, True])
    """
    if size < 0 or high < low:
        raise ValueError("size must be > 0 and high > low")
    dtype = akdtype(dtype) # normalize dtype
    # check dtype for error
    if dtype.name not in DTypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    lowstr = NUMBER_FORMAT_STRINGS[dtype.name].format(low)
    highstr = NUMBER_FORMAT_STRINGS[dtype.name].format(high)
    sizestr = NUMBER_FORMAT_STRINGS['int64'].format(size)

    repMsg = generic_msg(cmd='randint', args='{} {} {} {} {}'.\
                         format(sizestr, dtype.name, lowstr, highstr, seed))
    return create_pdarray(repMsg)

@typechecked
def uniform(size : int_scalars, low : numeric_scalars=float(0.0), 
            high : numeric_scalars=1.0, seed: Union[None, 
                                               int_scalars]=None) -> pdarray:
    """
    Generate a pdarray with uniformly distributed random float values 
    in a specified range.

    Parameters
    ----------
    low : float_scalars
        The low value (inclusive) of the range, defaults to 0.0
    high : float_scalars
        The high value (inclusive) of the range, defaults to 1.0
    size : int_scalars
        The length of the returned array
    seed : int_scalars, optional
        Value used to initialize the random number generator

    Returns
    -------
    pdarray, float64
        Values drawn uniformly from the specified range

    Raises
    ------
    TypeError
        Raised if dtype.name not in DTypes, size is not an int, or if
        either low or high is not an int or float
    ValueError
        Raised if size < 0 or if high < low

    Notes
    -----
    The logic for uniform is delegated to the ak.randint method which 
    is invoked with a dtype of float64

    Examples
    --------
    >>> ak.uniform(3)
    array([0.92176432277231968, 0.083130710959903542, 0.68894208386667544])

    >>> ak.uniform(size=3,low=0,high=5,seed=0)
    array([0.30013431967121934, 0.47383036230759112, 1.0441791878997098])
    """
    return randint(low=low, high=high, size=size, dtype='float64', seed=seed)

@typechecked
def standard_normal(size : int_scalars, seed : Union[None, int_scalars]=None) -> pdarray:
    """
    Draw real numbers from the standard normal distribution.

    Parameters
    ----------
    size : int_scalars
        The number of samples to draw (size of the returned array)
    seed : int_scalars
        Value used to initialize the random number generator
    
    Returns
    -------
    pdarray, float64
        The array of random numbers
        
    Raises
    ------
    TypeError
        Raised if size is not an int
    ValueError
        Raised if size < 0

    See Also
    --------
    randint

    Notes
    -----
    For random samples from :math:`N(\\mu, \\sigma^2)`, use:

    ``(sigma * standard_normal(size)) + mu``
    
    Examples
    --------
    >>> ak.standard_normal(3,1)
    array([-0.68586185091150265, 1.1723810583573375, 0.567584107142031])  
    """
    if size < 0:
        raise ValueError("The size parameter must be > 0")
    return create_pdarray(generic_msg(cmd='randomNormal', args='{} {}'.\
                    format(NUMBER_FORMAT_STRINGS['int64'].format(size), seed)))

@typechecked
def random_strings_uniform(minlen : int_scalars, maxlen : int_scalars, 
                        size : int_scalars, characters : str='uppercase', 
                           seed : Union[None, int_scalars]=None) -> Strings:
    """
    Generate random strings with lengths uniformly distributed between 
    minlen and maxlen, and with characters drawn from a specified set.

    Parameters
    ----------
    minlen : int_scalars
        The minimum allowed length of string
    maxlen : int_scalars
        The maximum allowed length of string
    size : int_scalars
        The number of strings to generate
    characters : (uppercase, lowercase, numeric, printable, binary)
        The set of characters to draw from
    seed :  Union[None, int_scalars], optional
        Value used to initialize the random number generator

    Returns
    -------
    Strings
        The array of random strings
        
    Raises
    ------
    ValueError
        Raised if minlen < 0, maxlen < minlen, or size < 0

    See Also
    --------
    random_strings_lognormal, randint
    
    Examples
    --------
    >>> ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=5)
    array(['TVKJ', 'EWAB', 'CO', 'HFMD', 'U'])
    
    >>> ak.random_strings_uniform(minlen=1, maxlen=5, seed=1, size=5, 
    ... characters='printable')
    array(['+5"f', '-P]3', '4k', '~HFF', 'F'])
    """
    if minlen < 0 or maxlen < minlen or size < 0:
        raise ValueError(("Incompatible arguments: minlen < 0, maxlen " +
                          "< minlen, or size < 0"))

    repMsg = generic_msg(cmd="randomStrings", args="{} {} {} {} {} {}".\
          format(NUMBER_FORMAT_STRINGS['int64'].format(size),
                 "uniform", characters,
                 NUMBER_FORMAT_STRINGS['int64'].format(minlen),
                 NUMBER_FORMAT_STRINGS['int64'].format(maxlen),
                 seed))
    return Strings(*(cast(str,repMsg).split('+')))

@typechecked
def random_strings_lognormal(logmean : numeric_scalars, logstd : numeric_scalars, 
            size : int_scalars, characters : str='uppercase', 
                             seed : Optional[int_scalars]=None) -> Strings:
    """
    Generate random strings with log-normally distributed lengths and 
    with characters drawn from a specified set.

    Parameters
    ----------
    logmean : numeric_scalars
        The log-mean of the length distribution
    logstd :  numeric_scalars
        The log-standard-deviation of the length distribution
    size : int_scalars
        The number of strings to generate
    characters : (uppercase, lowercase, numeric, printable, binary)
        The set of characters to draw from
    seed : int_scalars, optional
        Value used to initialize the random number generator

    Returns
    -------
    Strings
        The Strings object encapsulating a pdarray of random strings
    
    Raises
    ------
    TypeError
        Raised if logmean is neither a float nor a int, logstd is not a float, 
        size is not an int, or if characters is not a str
    ValueError
        Raised if logstd <= 0 or size < 0

    See Also
    --------
    random_strings_lognormal, randint

    Notes
    -----
    The lengths of the generated strings are distributed $Lognormal(\\mu, \\sigma^2)$,
    with :math:`\\mu = logmean` and :math:`\\sigma = logstd`. Thus, the strings will
    have an average length of :math:`exp(\\mu + 0.5*\\sigma^2)`, a minimum length of 
    zero, and a heavy tail towards longer strings.
    
    Examples
    --------
    >>> ak.random_strings_lognormal(2, 0.25, 5, seed=1)
    array(['TVKJTE', 'ABOCORHFM', 'LUDMMGTB', 'KWOQNPHZ', 'VSXRRL'])
    
    >>> ak.random_strings_lognormal(2, 0.25, 5, seed=1, characters='printable')
    array(['+5"fp-', ']3Q4kC~HF', '=F=`,IE!', 'DjkBa'9(', '5oZ1)='])
    """
    if not isSupportedNumber(logmean) or not isSupportedNumber(logstd):
        raise TypeError('both logmean and logstd must be an int, np.int64, float, or np.float64')
    if logstd <= 0 or size < 0:
        raise ValueError("Incompatible arguments: logstd <= 0 or size < 0")

    repMsg = generic_msg(cmd="randomStrings", args="{} {} {} {} {} {}".\
          format(NUMBER_FORMAT_STRINGS['int64'].format(size),
                 "lognormal", characters,
                 NUMBER_FORMAT_STRINGS['float64'].format(logmean),
                 NUMBER_FORMAT_STRINGS['float64'].format(logstd),
                 seed))
    return Strings(*(cast(str,repMsg).split('+')))
