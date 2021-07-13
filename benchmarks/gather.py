#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak

TYPES = ('int64', 'float64', 'bool')

def time_ak_gather(isize, vsize, trials, dtype, random, seed):
    print(">>> arkouda {} gather".format(dtype))
    cfg = ak.get_config()
    Ni = isize * cfg["numLocales"]
    Nv = vsize * cfg["numLocales"]
    print("numLocales = {}, num_indices = {:,} ; num_values = {:,}".format(cfg["numLocales"], Ni, Nv))
    # Index vector is always random
    i = ak.randint(0, Nv, Ni, seed=seed)
    if seed is not None:
        seed += 1
    if random or seed is not None:
        if dtype == 'int64':
            v = ak.randint(0, 2**32, Nv, seed=seed)
        elif dtype == 'float64':
            v = ak.randint(0, 1, Nv, dtype=ak.float64, seed=seed)
        elif dtype == 'bool':
            v = ak.randint(0, 1, Nv, dtype=ak.bool, seed=seed)
        elif dtype == 'str':
            v = ak.random_strings_uniform(1, 16, Nv, seed=seed)
    else:   
        if dtype == 'str':
            v = ak.cast(ak.arange(Nv), 'str')
        else:
            v = ak.ones(Nv, dtype=dtype)
    
    timings = []
    for _ in range(trials):
        start = time.time()
        c = v[i]
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    if dtype == 'str':
        offsets_transferred = 3 * c.size * 8
        bytes_transferred = (c.size * 8) + (2 * c.nbytes)
        bytes_per_sec = (offsets_transferred + bytes_transferred) / tavg
    else:
        bytes_per_sec = (c.size * c.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def time_np_gather(Ni, Nv, trials, dtype, random, seed):
    print(">>> numpy {} gather".format(dtype))
    print("num_indices = {:,} ; num_values = {:,}".format(Ni, Nv))
    # Index vector is always random
    if seed is not None:
        np.random.seed(seed)
    i = np.random.randint(0, Nv, Ni)
    if random or seed is not None:
        if dtype == 'int64':
            v = np.random.randint(0, 2**32, Nv)
        elif dtype == 'float64':
            v = np.random.random(Nv)
        elif dtype == 'bool':
            v = np.random.randint(0, 1, Nv, dtype=np.bool)
        elif dtype == 'str':
            v = np.array(np.random.randint(0, 2**32, Nv), dtype='str')
    else:   
        v = np.ones(Nv, dtype=dtype)
    
    timings = []
    for _ in range(trials):
        start = time.time()
        c = v[i]
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    bytes_per_sec = (c.size * c.itemsize * 3) / tavg
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def check_correctness(dtype, random, seed):
    Ni = 10**4
    Nv = 10**4
    if seed is not None:
        np.random.seed(seed)
    npi = np.random.randint(0, Nv, Ni)
    aki = ak.array(npi)
    if random or seed is not None:
        if dtype == 'int64':
            npv = np.random.randint(0, 2**32, Nv)
        elif dtype == 'float64':
            npv = np.random.random(Nv)
        elif dtype == 'bool':
            npv = np.random.randint(0, 1, Nv, dtype=np.bool)
        elif dtype == 'str':
            npv = np.array([np.str(x) for x in np.random.randint(0, 2**32, Nv)])
    else:   
        npv = np.ones(Nv, dtype=dtype)
    akv = ak.array(npv)
    npc = npv[npi]
    akc = akv[aki]
    if dtype == 'str':
        assert (npc == akc.to_ndarray()).all()
    else:
        assert np.allclose(npc, akc.to_ndarray())

def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of random gather: C = V[I]")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**8, help='Problem size: length of index and gather arrays')
    parser.add_argument('-i', '--index-size', type=int, help='Length of index array (number of gathers to perform)')
    parser.add_argument('-v', '--value-size', type=int, help='Length of array from which values are gathered')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='int64', help='Dtype of value array ({})'.format(', '.join(TYPES)))
    parser.add_argument('-r', '--randomize', default=False, action='store_true', help='Use random values instead of ones')
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Value to initialize random number generator')
    return parser
    
if __name__ == "__main__":
    import sys
    parser = create_parser()
    args = parser.parse_args()
    args.index_size = args.size if args.index_size is None else args.index_size
    args.value_size = args.size if args.value_size is None else args.value_size
    if args.dtype not in TYPES:
        raise ValueError("Dtype must be {}, not {}".format('/'.join(TYPES), args.dtype))
    ak.verbose = False
    ak.connect(args.hostname, args.port)

    if args.correctness_only:
        for dtype in TYPES:
            check_correctness(dtype, args.randomize, args.seed)
        sys.exit(0)
    
    print("size of index array = {:,}".format(args.index_size))
    print("size of values array = {:,}".format(args.value_size))
    print("number of trials = ", args.trials)
    time_ak_gather(args.index_size, args.value_size, args.trials, args.dtype, args.randomize, args.seed)
    if args.numpy:
        time_np_gather(args.index_size, args.value_size, args.trials, args.dtype, args.randomize, args.seed)
        print("Verifying agreement between arkouda and NumPy on small problem... ", end="")
        check_correctness(args.dtype, args.randomize, args.seed)
        print("CORRECT")
        
    sys.exit(0)
