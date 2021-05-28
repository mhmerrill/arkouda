#!/usr/bin/env python3

import time, argparse
import numpy as np
import arkouda as ak

TYPES = ('int64', 'float64')

def time_ak_coargsort(N_per_locale, trials, dtype, seed):
    print(">>> arkouda {} coargsort".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    for numArrays in (1, 2, 8, 16):
        if seed is None:
            seeds = [None for _ in range(numArrays)]
        else:
            seeds = [seed+i for i in range(numArrays)]
        if dtype == 'int64':
            arrs = [ak.randint(0, 2**32, N//numArrays, seed=s) for s in seeds]
            nbytes = sum(a.size * a.itemsize for a in arrs)
        elif dtype == 'float64':
            arrs = [ak.randint(0, 1, N//numArrays, dtype=ak.float64, seed=s) for s in seeds]
            nbytes = sum(a.size * a.itemsize for a in arrs)
        elif dtype == 'str':
            arrs = [ak.random_strings_uniform(1, 16, N//numArrays, seed=s) for s in seeds]
            nbytes = sum(a.nbytes * a.entry.itemsize for a in arrs)

        timings = []
        for i in range(trials):
            start = time.time()
            perm = ak.coargsort(arrs)
            end = time.time()
            timings.append(end - start)
        tavg = sum(timings) / trials

        a = arrs[0][perm]
        if dtype in ('int64', 'float64'):
            assert ak.is_sorted(a)
        print("{}-array Average time = {:.4f} sec".format(numArrays, tavg))
        bytes_per_sec = nbytes / tavg
        print("{}-array Average rate = {:.4f} GiB/sec".format(numArrays, bytes_per_sec/2**30))

def time_np_coargsort(N, trials, dtype, seed):
    print(">>> numpy {} coargsort".format(dtype)) # technically lexsort
    print("N = {:,}".format(N))
    if seed is not None:
        np.random.seed(seed)
    for numArrays in (1, 2, 8, 16):
        if dtype == 'int64':
            arrs = [np.random.randint(0, 2**32, N//numArrays) for _ in range(numArrays)]
        elif dtype == 'float64':
            arrs = [np.random.random(N//numArrays) for _ in range(numArrays)]
        elif dtype == 'str':
            arrs = [np.cast['str'](np.random.randint(0, 2**32, N//numArrays)) for _ in range(numArrays)]

        timings = []
        for i in range(trials):
            start = time.time()
            perm = np.lexsort(arrs)
            end = time.time()
            timings.append(end - start)
        tavg = sum(timings) / trials

        a = arrs[-1][perm]
        assert np.all(a[:-1] <= a[1:])

        print("{}-array Average time = {:.4f} sec".format(numArrays, tavg))
        bytes_per_sec = sum(a.size * a.itemsize for a in arrs) / tavg
        print("{}-array Average rate = {:.4f} GiB/sec".format(numArrays, bytes_per_sec/2**30))

def check_correctness(dtype, seed):
    N = 10**4
    if dtype == 'int64':
        a = ak.randint(0, 2**32, N, seed=seed)
        z = ak.zeros(N, dtype=dtype)
    elif dtype == 'float64':
        a = ak.randint(0, 1, N, dtype=ak.float64, seed=seed)
        z = ak.zeros(N, dtype=dtype)
    elif dtype == 'str':
        a = ak.random_strings_uniform(1, 16, N, seed=seed)
        z = ak.cast(ak.zeros(N), 'str')

    perm = ak.coargsort([a, z])
    if dtype in ('int64', 'float64'):
        assert ak.is_sorted(a[perm])
    perm = ak.coargsort([z, a])
    if dtype in ('int64', 'float64'):
        assert ak.is_sorted(a[perm])


def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of sorting arrays of random values.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: total length of all arrays to coargsort')
    parser.add_argument('-t', '--trials', type=int, default=1, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array ({})'.format(', '.join(TYPES)))
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Value to initialize random number generator')
    return parser

if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format('/'.join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype, args.seed)
        sys.exit(0)

    print("array size = {:,}".format(args.size))
    print("number of trials = ", args.trials)
    time_ak_coargsort(args.size, args.trials, args.dtype, args.seed)
    if args.numpy:
        time_np_coargsort(args.size, args.trials, args.dtype, args.seed)
    sys.exit(0)
