from typeguard import typechecked
import json, os, warnings
from typing import cast, Dict, List, Mapping, Optional, Union
from arkouda.client import generic_msg
from arkouda.pdarrayclass import pdarray, create_pdarray
from arkouda.strings import Strings

__all__ = ["ls_hdf", "read_hdf", "read_all", "load", "get_datasets",
           "load_all", "save_all"]

@typechecked
def ls_hdf(filename : str) -> str:
    """
    This function calls the h5ls utility on a filename visible to the
    arkouda server.

    Parameters
    ----------
    filename : str
        The name of the file to pass to h5ls

    Returns
    -------
    str
        The string output of `h5ls <filename>` from the server

    Raises
    ------
    TypeError
        Raised if filename is not a str
    ValueError
        Raised if filename is empty or contains only whitespace
    RuntimeError
        Raised if error occurs in executing ls on an HDF5 file
    """
    if not (filename and filename.strip()):
        raise ValueError("filename cannot be an empty string")

    return cast(str,generic_msg(cmd="lshdf", args="{}".format(json.dumps([filename]))))

@typechecked
def read_hdf(dsetName : str, filenames : Union[str,List[str]],
             strictTypes: bool=True) \
          -> Union[pdarray, Strings]:
    """
    Read a single dataset from multiple HDF5 files into an Arkouda
    pdarray or Strings object.

    Parameters
    ----------
    dsetName : str
        The name of the dataset (must be the same across all files)
    filenames : list or str
        Either a list of filenames or shell expression
    strictTypes: bool
        If True (default), require all dtypes in all files to have the
        same precision and sign. If False, allow dtypes of different
        precision and sign across different files. For example, if one 
        file contains a uint32 dataset and another contains an int64
        dataset, the contents of both will be read into an int64 pdarray.
        
    Returns
    -------
    Union[pdarray,Strings] 
        A pdarray or Strings instance pointing to the server-side data

    Raises
    ------
    TypeError 
        Raised if dsetName is not a str or if filenames is neither a string
        nor a list of strings
    ValueError 
        Raised if all datasets are not present in all hdf5 files  
    RuntimeError
        Raised if one or more of the specified files cannot be opened  

    See Also
    --------
    get_datasets, ls_hdf, read_all, load, save

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression
    (a single filename is a valid expression, so it will work) and is
    expanded with glob to read all matching files. Use ``get_datasets`` to
    show the names of datasets in HDF5 files.

    If dsetName is not present in all files, a TypeError is raised.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    # rep_msg = generic_msg("readhdf {} {:n} {}".format(dsetName, len(filenames), json.dumps(filenames)))
    # # This is a hack to detect a string return type
    # # In the future, we should put the number and type into the return message
    # if '+' in rep_msg:
    #     return Strings(*rep_msg.split('+'))
    # else:
    #     return create_pdarray(rep_msg)
    return cast(Union[pdarray, Strings], 
                read_all(filenames, datasets=dsetName, strictTypes=strictTypes))


def read_all(filenames : Union[str, List[str]],
             datasets: Optional[Union[str, List[str]]] = None,
             iterative: bool = False,
             strictTypes: bool = True,
             allow_errors: bool = False)\
             -> Union[pdarray, Strings, Mapping[str,Union[pdarray,Strings]]]:
    """
    Read datasets from HDF5 files.

    Parameters
    ----------
    filenames : list or str
        Either a list of filenames or shell expression
    datasets : list or str or None
        (List of) name(s) of dataset(s) to read (default: all available)
    iterative : bool
        Iterative (True) or Single (False) function call(s) to server
    strictTypes: bool
        If True (default), require all dtypes of a given dataset to have the
        same precision and sign. If False, allow dtypes of different
        precision and sign across different files. For example, if one 
        file contains a uint32 dataset and another contains an int64
        dataset with the same name, the contents of both will be read 
        into an int64 pdarray.
    allow_errors: bool
        Default False, if True will allow files with read errors to be skipped
        instead of failing.  A warning will be included in the return containing
        the total number of files skipped due to failure and up to 10 filenames.

    Returns
    -------
    For a single dataset returns an Arkouda pdarray or Arkouda Strings object
    and for multiple datasets returns a dictionary of Arkouda pdarrays or
    Arkouda Strings.
        Dictionary of {datasetName: pdarray or String}

    Raises
    ------
    ValueError 
        Raised if all datasets are not present in all hdf5 files or if one or
        more of the specified files do not exist
    RuntimeError
        Raised if one or more of the specified files cannot be opened.
        If `allow_errors` is true this may be raised if no values are returned
        from the server.
    TypeError
        Raised if we receive an unknown arkouda_type returned from the server

    See Also
    --------
    read_hdf, get_datasets, ls_hdf

    Notes
    -----
    If filenames is a string, it is interpreted as a shell expression
    (a single filename is a valid expression, so it will work) and is
    expanded with glob to read all matching files.

    If iterative == True each dataset name and file names are passed to
    the server as independent sequential strings while if iterative == False
    all dataset names and file names are passed to the server in a single
    string.

    If datasets is None, infer the names of datasets from the first file
    and read all of them. Use ``get_datasets`` to show the names of datasets
    to HDF5 files.
    """
    if isinstance(filenames, str):
        filenames = [filenames]
    if datasets is None:
        datasets = get_datasets_allow_errors(filenames) if allow_errors else get_datasets(filenames[0])
    if isinstance(datasets, str):
        datasets = [datasets]
    else:  # ensure dataset(s) exist
        if isinstance(datasets, str):
            datasets = [datasets]
        nonexistent = set(datasets) - \
            (set(get_datasets_allow_errors(filenames)) if allow_errors else set(get_datasets(filenames[0])))
        if len(nonexistent) > 0:
            raise ValueError("Dataset(s) not found: {}".format(nonexistent))
    if iterative == True: # iterative calls to server readhdf
        return {dset:read_hdf(dset, filenames, strictTypes=strictTypes) for dset in datasets}
    else:  # single call to server readAllHdf
        rep_msg = generic_msg(cmd="readAllHdf", args="{} {:n} {:n} {} {} | {}".
                              format(strictTypes, len(datasets), len(filenames), allow_errors, json.dumps(datasets),
                                     json.dumps(filenames)))
        rep = json.loads(rep_msg)  # See GenSymIO._buildReadAllHdfMsgJson for json structure
        items = rep["items"] if "items" in rep else []
        file_errors = rep["file_errors"] if "file_errors" in rep else []
        if allow_errors and file_errors:
            file_error_count = rep["file_error_count"] if "file_error_count" in rep else -1
            warnings.warn(f"There were {file_error_count} errors reading files on the server. " +
                          f"Sample error messages {file_errors}", RuntimeWarning)

        # We have a couple possible return conditions
        # 1. We have multiple items returned i.e. multi pdarrays, multi strings, multi pdarrays & strings
        # 2. We have a single pdarray
        # 3. We have a single strings object
        if len(items) > 1: #  DataSets condition
            d: Dict[str, Union[pdarray, Strings]] = {}
            for item in items:
                if "seg_string" == item["arkouda_type"]:
                    d[item["dataset_name"]] = Strings.from_parts(*item["created"].split("+"))
                elif "pdarray" == item["arkouda_type"]:
                    d[item["dataset_name"]] = create_pdarray(item["created"])
                else:
                    raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
            return d
        elif len(items) == 1:
            item = items[0]
            if "pdarray" == item["arkouda_type"]:
                return create_pdarray(item["created"])
            elif "seg_string" == item["arkouda_type"]:
                return Strings.from_return_msg(item["created"])
            else:
                raise TypeError(f"Unknown arkouda type:{item['arkouda_type']}")
        else:
            raise RuntimeError("No items were returned")


@typechecked
def load(path_prefix : str, dataset : str='array') -> Union[pdarray,Strings]:
    """
    Load a pdarray previously saved with ``pdarray.save()``.

    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray
    dataset : str
        Dataset name where the pdarray was saved, defaults to 'array'

    Returns
    -------
    Union[pdarray, Strings]
        The pdarray or Strings that was previously saved

    Raises
    ------
    TypeError 
        Raised if either path_prefix or dataset is not a str 
    ValueError 
        Raised if the dataset is not present in all hdf5 files or if the
        path_prefix does not correspond to files accessible to Arkouda
    RuntimeError
        Raised if the hdf5 files are present but there is an error in opening
        one or more of them

    See Also
    --------
    save, load_all, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    globstr = "{}_LOCALE*{}".format(prefix, extension)

    try:
        return read_hdf(dataset, globstr)
    except RuntimeError as re:
        if 'does not exist' in str(re):
            raise ValueError('There are no files corresponding to the ' +
                                'path_prefix {} in location accessible to Arkouda'.format(prefix))
        else:
            raise RuntimeError(re)
            

@typechecked
def get_datasets(filename : str) -> List[str]:
    """
    Get the names of datasets in an HDF5 file.

    Parameters
    ----------
    filename : str
        Name of an HDF5 file visible to the arkouda server

    Returns
    -------
    List[str]
        Names of the datasets in the file
        
    Raises
    ------
    TypeError
        Raised if filename is not a str
    ValueError
        Raised if filename is empty or contains only whitespace
    RuntimeError
        Raised if error occurs in executing ls on an HDF5 file

    See Also
    --------
    ls_hdf
    """
    rep_msg = ls_hdf(filename)
    datasets = [line.split()[0] for line in rep_msg.splitlines()]
    return datasets

@typechecked
def get_datasets_allow_errors(filenames: List[str]) -> List[str]:
    """
    Get the names of datasets in an HDF5 file
    Allow file read errors until success

    Parameters
    ----------
    filenames : List[str]
        A list of HDF5 files visible to the arkouda server

    Returns
    -------
    List[str]
        Names of the datasets in the file

    Raises
    ------
    TypeError
        Raised if filenames is not a List[str]
    FileNotFoundError
        If none of the files could be read successfully

    See Also
    --------
    get_datasets, ls_hdf
    """
    datasets = []
    for filename in filenames:
        try:
            datasets = get_datasets(filename)
            break
        except RuntimeError:
            pass
    if not datasets:  # empty
        raise FileNotFoundError("Could not read any of the requested files")
    return datasets

@typechecked
def load_all(path_prefix : str) -> Mapping[str,Union[pdarray,Strings]]:
    """
    Load multiple pdarrays or Strings previously saved with ``save_all()``.

    Parameters
    ----------
    path_prefix : str
        Filename prefix used to save the original pdarray

    Returns
    -------
    Mapping[str,pdarray]
        Dictionary of {datsetName: pdarray} with the previously saved pdarrays
        
        
    Raises
    ------
    TypeError:
        Raised if path_prefix is not a str
    ValueError 
        Raised if all datasets are not present in all hdf5 files or if the
        path_prefix does not correspond to files accessible to Arkouda   
    RuntimeError
        Raised if the hdf5 files are present but there is an error in opening
        one or more of them

    See Also
    --------
    save_all, load, read_hdf, read_all
    """
    prefix, extension = os.path.splitext(path_prefix)
    firstname = "{}_LOCALE0000{}".format(prefix, extension)
    try:
        return {dataset: load(path_prefix, dataset=dataset) \
                                       for dataset in get_datasets(firstname)}
    except RuntimeError as re:
        # enables backwards compatibility with previous naming convention
        if 'does not exist' in str(re):
            try: 
                firstname = "{}_LOCALE0{}".format(prefix, extension)
                return {dataset: load(path_prefix, dataset=dataset) \
                                       for dataset in get_datasets(firstname)}
            except RuntimeError as re:
                if 'does not exist' in str(re):
                    raise ValueError('There are no files corresponding to the ' +
                                     'path_prefix {} in location accessible to Arkouda'.format(prefix))
                else:
                    raise RuntimeError(re)
        else:
            raise RuntimeError('Could not open on or more files with ' +
                                   'the file prefix {}, check file format or permissions'.format(prefix))

def save_all(columns : Union[Mapping[str,pdarray],List[pdarray]], prefix_path : str, 
             names : List[str]=None, mode : str='truncate') -> None:
    """
    Save multiple named pdarrays to HDF5 files.

    Parameters
    ----------
    columns : dict or list of pdarrays
        Collection of arrays to save
    prefix_path : str
        Directory and filename prefix for output files
    names : list of str
        Dataset names for the pdarrays
    mode : {'truncate' | 'append'}
        By default, truncate (overwrite) the output files if they exist.
        If 'append', attempt to create new dataset in existing files.

    Returns
    -------
    None

    Raises
    ------
    ValueError 
        Raised if (1) the lengths of columns and values differ or (2) the mode 
        is not 'truncate' or 'append'

    See Also
    --------
    save, load_all

    Notes
    -----
    Creates one file per locale containing that locale's chunk of each pdarray.
    If columns is a dictionary, the keys are used as the HDF5 dataset names.
    Otherwise, if no names are supplied, 0-up integers are used. By default,
    any existing files at path_prefix will be overwritten, unless the user
    specifies the 'append' mode, in which case arkouda will attempt to add
    <columns> as new datasets to existing files. If the wrong number of files
    is present or dataset names already exist, a RuntimeError is raised.
    """
    if names is not None:
        if len(names) != len(columns):
            raise ValueError("Number of names does not match number of columns")
        else:
            datasetNames = names
    if isinstance(columns, dict):
        pdarrays = list(columns.values())
        if names is None:
            datasetNames = list(columns.keys())
    elif isinstance(columns, list):
        pdarrays = cast(List[pdarray],columns)
        if names is None:
            datasetNames = [str(column) for column in range(len(columns))]
    if (mode.lower() not in 'append') and (mode.lower() not in 'truncate'):
        raise ValueError("Allowed modes are 'truncate' and 'append'")
    first_iter = True
    for arr, name in zip(pdarrays, cast(List[str], datasetNames)):
        '''Append all pdarrays to existing files as new datasets EXCEPT the first one, 
           and only if user requests truncation'''
        if mode.lower() not in 'append' and first_iter:
            arr.save(prefix_path=prefix_path, dataset=name, mode='truncate')
            first_iter = False
        else:
            arr.save(prefix_path=prefix_path, dataset=name, mode='append')
