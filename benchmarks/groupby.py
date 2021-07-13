#!/usr/bin/env python3

import time, argparse
import numpy as np
import arkouda as ak

TYPES = ('int64',)

def generate_arrays(N, numArrays, dtype, seed):
    totalbytes = 0
    arrays = []
    for i in range(numArrays):
        if dtype == 'int64' or (i % 2 == 0 and dtype == 'mixed'):
            a = ak.randint(0, 2**32, N//numArrays, seed=seed)
            arrays.append(a)
            totalbytes += a.size * a.itemsize
        else:
            a = ak.random_strings_uniform(1, 16, N//numArrays, seed=seed)
            arrays.append(a)
            totalbytes += (a.nbytes * a.entry.itemsize)
        if seed is not None:
            seed += 1
    if numArrays == 1:
        arrays = arrays[0]
    return arrays, totalbytes

def time_ak_groupby(N_per_locale, trials, dtype, seed):
    print(">>> arkouda {} groupby".format(dtype))
    cfg = ak.get_config()
    N = N_per_locale * cfg["numLocales"]
    print("numLocales = {}, N = {:,}".format(cfg["numLocales"], N))
    for numArrays in (1, 2, 8, 16):
        if dtype == "mixed" and numArrays == 1:
            continue
        arrays, totalbytes = generate_arrays(N, numArrays, dtype, seed)
        timings = []
        for i in range(trials):
            start = time.time()
            g = ak.GroupBy(arrays)
            end = time.time()
            timings.append(end - start)
        tavg = sum(timings) / trials
        print("{}-array Average time = {:.4f} sec".format(numArrays, tavg))
        bytes_per_sec = totalbytes / tavg
        print("{}-array Average rate = {:.4f} GiB/sec".format(numArrays, bytes_per_sec/2**30))

def check_correctness(dtype, seed):
    arrays, totalbytes = generate_arrays(1000, 2, dtype, seed)
    g = ak.GroupBy(arrays)
    perm = ak.argsort(ak.randint(0, 2**32, arrays[0].size))
    g2 = ak.GroupBy([a[perm] for a in arrays])
    assert all((uk == uk2).all() for uk, uk2 in zip(g.unique_keys, g2.unique_keys))
    assert (g.segments == g2.segments).all()

def create_parser():
    parser = argparse.ArgumentParser(description="Measure performance of grouping arrays of random values.")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: total length of all arrays to group')
    parser.add_argument('-t', '--trials', type=int, default=1, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of array ({})'.format(', '.join(TYPES)))
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
    time_ak_groupby(args.size, args.trials, args.dtype, args.seed)
    sys.exit(0)

