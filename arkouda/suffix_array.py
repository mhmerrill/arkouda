from __future__ import annotations

from typing import cast, Union
from typeguard import typechecked
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray, _parse_single_int_array_value
from arkouda.strings import Strings
from arkouda.logger import getArkoudaLogger
import numpy as np  # type: ignore
from arkouda.dtypes import resolve_scalar_dtype, translate_np_dtype, int64
import json

__all__ = ["SArrays", "suffix_array", "lcp_array", "suffix_array_file"]


class SArrays:
    """
    Represents an array of (suffix) arrays whose data resides on the arkouda server.
    The user should not call this class directly; rather its instances are created
    by other arkouda functions. It is very similar to Strings and the difference is
    that its content is int arrays instead of strings.

    Attributes
    ----------
    offsets : pdarray
        The starting indices for each suffix array
    bytes : pdarray
        The raw integer indices of all suffix arrays
    size : int
        The number of suffix arrays in the array
    nbytes : int
        The total number of indices in all suffix arrays
        We have the same number indices as the number of characters/suffixes in strings
    ndim : int
        The rank of the array (currently only rank 1 arrays supported)
    shape : tuple
        The sizes of each dimension of the array
    dtype : dtype
        The dtype is np.int
    logger : ArkoudaLogger
        Used for all logging operations

    Notes
    -----
    SArrays is composed of two pdarrays: (1) offsets, which contains the
    starting indices for each string's suffix array  and (2) bytes, which contains the
    indices of all suffix arrays, no any spliter between two index arrays.
    """

    BinOps = frozenset(["==", "!="])
    objtype = "int"

    def __init__(
        self,
        offset_attrib: Union[pdarray, np.ndarray],
        bytes_attrib: Union[pdarray, np.ndarray],
    ) -> None:
        """
        Initializes the SArrays instance by setting all instance
        attributes, some of which are derived from the array parameters.

        Parameters
        ----------
        offset_attrib : Union[pdarray, np.ndarray,array]
            the array containing the offsets
        bytes_attrib : Union[pdarray, np.ndarray,array]
            the array containing the suffix array indices

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            Raised if there's an error converting a Numpy array or standard
            Python array to either the offset_attrib or bytes_attrib
        ValueError
            Raised if there's an error in generating instance attributes
            from either the offset_attrib or bytes_attrib parameter
        """
        if isinstance(offset_attrib, pdarray):
            self.offsets = offset_attrib
        else:
            try:
                self.offsets = create_pdarray(offset_attrib)
            except Exception as e:
                raise RuntimeError(e)
        if isinstance(bytes_attrib, pdarray):
            self.bytes = bytes_attrib
        else:
            try:
                self.bytes = create_pdarray(bytes_attrib)
            except Exception as e:
                raise RuntimeError(e)
        try:
            self.size = self.offsets.size
            self.nbytes = self.bytes.size
            self.ndim = self.offsets.ndim
            self.shape = self.offsets.shape
        except Exception as e:
            raise ValueError(e)
        self.dtype = int64
        self.logger = getArkoudaLogger(name=__class__.__name__)  # type: ignore

    def __iter__(self):
        raise NotImplementedError("SArrays does not support iteration now")

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            vals = ["'{}'".format(self[i]) for i in range(self.size)]
        else:
            vals = ["'{}'".format(self[i]) for i in range(3)]
            vals.append("... ")
            vals.extend([self[i] for i in range(self.size - 3, self.size)])
        return "[{}]".format(", ".join(vals))

    def __repr__(self) -> str:
        return "array({})".format(self.__str__())

    @typechecked
    def _binop(self, other: Union[SArrays, np.int_], op: str) -> pdarray:
        """
        Executes the requested binop on this SArrays instance and the
        parameter SArrays object and returns the results within
        a pdarray object.

        Parameters
        ----------
        other : SArrays
            the other object is a SArrays object
        op : str
            name of the binary operation to be performed

        Returns
        -------
        pdarray
            encapsulating the results of the requested binop

        Raises
        -----
        ValueError
            Raised if (1) the op is not in the self.BinOps set, or (2) if the
            sizes of this and the other instance don't match, or (3) the other
            object is not a SArrays object
        RuntimeError
            Raised if a server-side error is thrown while executing the
            binary operation
        """
        if op not in self.BinOps:
            raise ValueError("SArrays: unsupported operator: {}".format(op))
        if isinstance(other, Strings):
            if self.size != other.size:
                raise ValueError(
                    "SArrays: size mismatch {} {}".format(self.size, other.size)
                )
            cmd = "segmentedBinopvvInt"
            args = "{} {} {} {} {} {} {}".format(
                op,
                self.objtype,
                self.offsets.name,
                self.bytes.name,
                other.objtype,
                other.offsets.name,
                other.bytes.name,
            )
        elif resolve_scalar_dtype(other) == "int":
            cmd = "segmentedBinopvsInt"
            args = "{} {} {} {} {} {}".format(
                op,
                self.objtype,
                self.offsets.name,
                self.bytes.name,
                self.objtype,
                json.dumps([other]),
            )
        else:
            raise ValueError(
                "SArrays: {} not supported between SArrays and {}".format(
                    op, other.__class__.__name__
                )
            )
        repMsg = generic_msg(cmd=cmd, args=args)
        return create_pdarray(cast(str, repMsg))

    def __eq__(self, other) -> bool:
        return self._binop(other, "==")

    def __ne__(self, other) -> bool:
        return self._binop(cast(SArrays, other), "!=")

    def __getitem__(self, key):
        if np.isscalar(key) and resolve_scalar_dtype(key) == "int64":
            orig_key = key
            if key < 0:
                # Interpret negative key as offset from end of array
                key += self.size
            if key >= 0 and key < self.size:
                cmd = "segmentedIndex"
                args = "{} {} {} {} {}".format(
                    "intIndex", self.objtype, self.offsets.name, self.bytes.name, key
                )
                repMsg = generic_msg(cmd=cmd, args=args)
                _, value = repMsg.split(maxsplit=1)
                return _parse_single_int_array_value(value)
            else:
                raise IndexError(
                    "[int] {} is out of bounds with size {}".format(orig_key, self.size)
                )
        elif isinstance(key, slice):
            (start, stop, stride) = key.indices(self.size)
            self.logger.debug(
                "start: {}; stop: {}; stride: {}".format(start, stop, stride)
            )
            cmd = "segmentedIndex"
            args = "{} {} {} {} {} {} {}".format(
                "sliceIndex",
                self.objtype,
                self.offsets.name,
                self.bytes.name,
                start,
                stop,
                stride,
            )
            repMsg = generic_msg(cmd=cmd, args=args)
            offsets, values = repMsg.split("+")
            return SArrays(offsets, values)
        elif isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("bool", "int"):
                raise TypeError("unsupported pdarray index type {}".format(key.dtype))
            if kind == "int" and self.size != key.size:
                raise ValueError("size mismatch {} {}".format(self.size, key.size))
            cmd = "segmentedIndex"
            args = "{} {} {} {} {}".format(
                "pdarrayIndex",
                self.objtype,
                self.offsets.name,
                self.bytes.name,
                key.name,
            )
            repMsg = generic_msg(cmd=cmd, args=args)
            offsets, values = repMsg.split("+")
            return SArrays(offsets, values)
        else:
            raise TypeError(
                "unsupported pdarray index type {}".format(key.__class__.__name__)
            )

    def get_lengths(self) -> pdarray:
        """
        Return the length of each suffix array in the array.

        Returns
        -------
        pdarray, int
            The length of each string

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown
        """
        cmd = "segmentLengths"
        args = "{} {} {}".format(self.objtype, self.offsets.name, self.bytes.name)
        repMsg = generic_msg(cmd=cmd, args=args)
        return create_pdarray(cast(str, repMsg))

    def save(
        self, prefix_path: str, dataset: str = "int_array", mode: str = "truncate"
    ) -> None:
        """
        Save the SArrays object to HDF5. The result is a collection of HDF5 files,
        one file per locale of the arkouda server, where each filename starts
        with prefix_path. Each locale saves its chunk of the array to its
        corresponding file.

        Parameters
        ----------
        prefix_path : str
            Directory and filename prefix that all output files share
        dataset : str
            The name of the SArrays dataset to be written, defaults to int_array
        mode : str {'truncate' | 'append'}
            By default, truncate (overwrite) output files, if they exist.
            If 'append', create a new SArrays dataset within existing files.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Raised if the lengths of columns and values differ, or the mode is
            neither 'truncate' nor 'append'

        See Also
        --------
        pdarrayIO.save

        Notes
        -----
        Important implementation notes: (1) SArrays state is saved as two datasets
        within an hdf5 group, (2) the hdf5 group is named via the dataset parameter,
        (3) the hdf5 group encompasses the two pdarrays composing a SArrays object:
        segments and values and (4) save logic is delegated to pdarray.save
        """
        self.bytes.save(
            prefix_path=prefix_path, dataset="{}/values".format(dataset), mode=mode
        )

    @classmethod
    def register_helper(cls, offsets, bytes):
        return cls(offsets, bytes)

    def register(self, user_defined_name: str) -> "SArrays":
        return self.register_helper(
            self.offsets.register(user_defined_name + "_offsets"),
            self.bytes.register(user_defined_name + "_bytes"),
        )

    def unregister(self) -> None:
        self.offsets.unregister()
        self.bytes.unregister()

    @staticmethod
    def attach(user_defined_name: str) -> "SArrays":
        return SArrays(
            pdarray.attach(user_defined_name + "_offsets"),
            pdarray.attach(user_defined_name + "_bytes"),
        )


@typechecked
def suffix_array(strings: Strings) -> SArrays:
    """
        Return the suffix arrays of given strings. The size/shape of each suffix
    arrays is the same as the corresponding strings.
    A simple example of suffix array is as follow. Given a string "banana$",
    all the suffixes are as follows.
    s[0]="banana$"
    s[1]="anana$"
    s[2]="nana$"
    s[3]="ana$"
    s[4]="na$"
    s[5]="a$"
    s[6]="$"
    The suffix array of string "banana$"  is the array of indices of sorted suffixes.
    s[6]="$"
    s[5]="a$"
    s[3]="ana$"
    s[1]="anana$"
    s[0]="banana$"
    s[4]="na$"
    s[2]="nana$"
    so sa=[6,5,3,1,0,4,2]

        Returns
        -------
        pdarray
            The suffix arrays of the given strings

        See Also
        --------

        Notes
        -----

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error in executing group request or
            creating the pdarray encapsulating the return message
    """
    cmd = "segmentedSuffixAry"
    args = "{} {} {}".format(strings.objtype, strings.offsets.name, strings.bytes.name)
    repMsg = generic_msg(cmd=cmd, args=args)
    return SArrays(*(cast(str, repMsg).split("+")))


@typechecked
def lcp_array(suffixarrays: SArrays, strings: Strings) -> SArrays:
    """
        Return the longest common prefix of given suffix arrays. The size/shape of each lcp
    arrays is the same as the corresponding suffix array.
        -------
        SArrays
            The LCP arrays of the given suffix arrays

        See Also
        --------

        Notes
        -----

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error in executing group request or
            creating the pdarray encapsulating the return message
    """
    cmd = "segmentedLCP"
    args = "{} {} {} {} {}".format(
        suffixarrays.objtype,
        suffixarrays.offsets.name,
        suffixarrays.bytes.name,
        strings.offsets.name,
        strings.bytes.name,
    )
    repMsg = generic_msg(cmd=cmd, args=args)
    return SArrays(*(cast(str, repMsg).split("+")))


@typechecked
def suffix_array_file(filename: str) -> tuple:
    """
        This function is major used for testing correctness and performance
        Return the suffix array of given file name's content as a string.
    A simple example of suffix array is as follow. Given string "banana$",
    all the suffixes are as follows.
    s[0]="banana$"
    s[1]="anana$"
    s[2]="nana$"
    s[3]="ana$"
    s[4]="na$"
    s[5]="a$"
    s[6]="$"
    The suffix array of string "banana$"  is the array of indices of sorted suffixes.
    s[6]="$"
    s[5]="a$"
    s[3]="ana$"
    s[1]="anana$"
    s[0]="banana$"
    s[4]="na$"
    s[2]="nana$"
    so sa=[6,5,3,1,0,4,2]

        Returns
        -------
        pdarray
            The suffix arrays of the given strings

        See Also
        --------

        Notes
        -----

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error in executing group request or
            creating the pdarray encapsulating the return message
    """
    cmd = "segmentedSAFile"
    args = "{}".format(filename)
    repMsg = generic_msg(cmd=cmd, args=args)
    tmpmsg = cast(str, repMsg).split("+")
    sastr = tmpmsg[0:2]
    strstr = tmpmsg[2:4]
    suffixarray = SArrays(*(cast(str, sastr)))
    originalstr = Strings(*(cast(str, strstr)))
    return suffixarray, originalstr
