from typing import cast, Tuple, Union
from enum import Enum
import numpy as np # type: ignore
from typeguard import typechecked
import builtins
import sys

__all__ = ["DTypes", "DTypeObjects", "dtype", "bool", "int64", "float64", 
           "uint8", "str_", "check_np_dtype", "translate_np_dtype", 
           "resolve_scalar_dtype", "ARKOUDA_SUPPORTED_DTYPES", "bool_scalars",
           "float_scalars", "int_scalars", "numeric_scalars", "numpy_scalars",
           "str_scalars", "all_scalars", "get_byteorder",
           "get_server_byteorder"]

NUMBER_FORMAT_STRINGS = {'bool': '{}',
                         'int64': '{:n}',
                         'float64': '{:.17f}',
                         'uint8': '{:n}',
                         'np.float64': 'f'}

dtype = np.dtype
bool = np.dtype(np.bool)
int64 = np.dtype(np.int64)
float64 = np.dtype(np.float64)
uint8 = np.dtype(np.uint8)
str_ = np.dtype(np.str_)
npstr = np.dtype(np.str)

# Union aliases used for static and runtime type checking
bool_scalars = Union[builtins.bool,np.bool]
float_scalars = Union[float,np.float64]
int_scalars = Union[int,np.int64]
numeric_scalars = Union[float,np.float64,int,np.int64]
numpy_scalars = Union[np.float64,np.int64,np.bool,np.uint8,np.str,np.str_]
str_scalars = Union[str, np.str, np.str_]
all_scalars = Union[float,np.float64,int,np.int64,
                                  builtins.bool,np.bool,str,np.str,np.str_]

'''
The DType enum defines the supported Arkouda data types in string form.
'''
class DType(Enum):
    
    BOOL = 'bool'
    FLOAT = 'float'
    FLOAT64 = 'float64'
    INT = 'int'
    INT64 = 'int64'
    STR = 'str'
    UINT8 = 'uint8'
    
    def __str__(self) -> str: # type: ignore
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter
        """
        return self.value
    
    def __repr__(self) -> str: # type: ignore
        """
        Overridden method returns value, which is useful in outputting
        a DType as a request parameter
        """
        return self.value

ARKOUDA_SUPPORTED_INTS = (int,np.int64)
ARKOUDA_SUPPORTED_FLOATS = (float,np.float64)
ARKOUDA_SUPPORTED_NUMBERS = (int,np.int64,float,np.float64)
ARKOUDA_SUPPORTED_DTYPES = frozenset([member.value for _, 
                                      member in DType.__members__.items()])

DTypes = frozenset([member.value for _, member in DType.__members__.items()])
DTypeObjects = frozenset([bool, float, float64, int, int64, str, str_, uint8])
NumericDTypes = frozenset(['bool', 'float', 'float64', 'int', 'int64'])
SeriesDTypes = {'string' : np.str_,
                 "<class 'str'>" : np.str_,
                 'int64' : np.int64,
                 "<class 'numpy.int64'>" : np.int64,                
                 'float64' : np.float64,
                 "<class 'numpy.float64'>" : np.float64,                   
                 'bool' : np.bool,
                 "<class 'bool'>" : np.bool,
                 'datetime64[ns]' : np.int64,
                 'timedelta64[ns]' : np.int64
                }

def isSupportedInt(num):
    return isinstance(num, ARKOUDA_SUPPORTED_INTS)

def isSupportedFloat(num):
    return isinstance(num, ARKOUDA_SUPPORTED_FLOATS)

def isSupportedNumber(num):
    return isinstance(num, ARKOUDA_SUPPORTED_NUMBERS)

def _as_dtype(dt) -> np.dtype:
    if not isinstance(dt, np.dtype):
        return np.dtype(dt)
    return dt

@typechecked
def check_np_dtype(dt : np.dtype) -> None:
    """
    Assert that numpy dtype dt is one of the dtypes supported
    by arkouda, otherwise raise TypeError.
    
    Raises
    ------
    TypeError
        Raised if the dtype is not in supported dtypes or if
        dt is not a np.dtype
    """
    
    if _as_dtype(dt).name not in DTypes:
        raise TypeError("Unsupported type: {}".format(dt))

@typechecked
def translate_np_dtype(dt: np.dtype) -> Tuple[builtins.str, int]:
    """
    Split numpy dtype dt into its kind and byte size, raising
    TypeError for unsupported dtypes.
    
    Raises
    ------
    TypeError
        Raised if the dtype is not in supported dtypes or if
        dt is not a np.dtype
    """
    # Assert that dt is one of the arkouda supported dtypes
    dt = _as_dtype(dt)
    check_np_dtype(dt)
    trans = {'i': 'int', 'f': 'float', 'b': 'bool', 
             'u': 'uint', 'U' : 'str'}
    kind = trans[dt.kind]
    return kind, dt.itemsize

def resolve_scalar_dtype(val : object) -> str: # type: ignore
    """
    Try to infer what dtype arkouda_server should treat val as.
    """
    # Python bool or np.bool
    if isinstance(val, builtins.bool) or (hasattr(val, 'dtype') \
                                and cast(np.bool,val).dtype.kind == 'b'):
        return 'bool'
    # Python int or np.int* or np.uint*
    elif isinstance(val, int) or (hasattr(val, 'dtype') and \
                                  cast(np.uint,val).dtype.kind in 'ui'):
        return 'int64'
    # Python float or np.float*
    elif isinstance(val, float) or (hasattr(val, 'dtype') and \
                                    cast(np.float, val).dtype.kind == 'f'):
        return 'float64'
    elif isinstance(val, builtins.str) or isinstance(val, np.str):
        return 'str'
    # Other numpy dtype
    elif hasattr(val, 'dtype'):
        return cast(np.dtype, val).dtype.name
    # Other python type
    else:
        return builtins.str(type(val))

def get_byteorder(dt: np.dtype) -> str:
    """
    Get a concrete byteorder (turns '=' into '<' or '>')
    """
    if dt.byteorder == '=':
        if sys.byteorder == 'little':
            return '<'
        elif sys.byteorder == 'big':
            return '>'
        else:
            raise ValueError("Client byteorder must be 'little' or 'big'")
    else:
        return dt.byteorder

def get_server_byteorder() -> str:
    """
    Get the server's byteorder
    """
    from arkouda.client import get_config
    order = get_config()['byteorder']
    if order not in ('little', 'big'):
        raise ValueError("Server byteorder must be 'little' or 'big'")
    return cast('str', order)
