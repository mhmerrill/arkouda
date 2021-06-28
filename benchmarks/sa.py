#!/usr/bin/env python3                                                         

import time, argparse
import numpy as np
import arkouda as ak
import random
import string

TYPES = ('int64', 'float64', 'bool', 'str')

def time_ak_sa( vsize,strlen, trials, dtype):
    print(">>> arkouda suffix array")
    cfg = ak.get_config()
    Nv = vsize * cfg["numLocales"]
    print("numLocales = {},  num of strings  = {:,}".format(cfg["numLocales"], Nv))

    if dtype == 'str':
         v = ak.random_strings_uniform(1, strlen, Nv)
    else:
        print("Wrong data type")
    c=ak.suffix_array(v)
#    print("size of suffix array={}".format(c.bytes.size))    
#    print("offset/number of suffix array={}".format(c.offsets.size))    
#    print("itemsize of suffix array={}".format(c.offsets.itemsize))    
    print("All the random strings are as follows")
    for k in range(vsize):
       print("the {} th random tring ={}".format(k,v[k]))    
       print("the {} th suffix array ={}".format(k,c[k]))    
       print("")
    timings = []
    for _ in range(trials):
        start = time.time()
        c=ak.suffix_array(v)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials

    print("Average time = {:.4f} sec".format(tavg))
    if dtype == 'str':
        offsets_transferred = 0 * c.offsets.size * c.offsets.itemsize
        bytes_transferred = (c.bytes.size * c.offsets.itemsize) + (0 * c.bytes.size)
        bytes_per_sec = (offsets_transferred + bytes_transferred) / tavg
    else:
        print("Wrong data type")
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))


def suffixArray(s):
    suffixes = [(s[i:], i) for i in range(len(s))]
    suffixes.sort(key=lambda x: x[0])
    sa= [s[1] for s in suffixes]
    #sa.insert(0,len(sa))
    return sa

def time_np_sa(vsize, strlen, trials, dtype):
    s=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(strlen))
    timings = []
    for _ in range(trials):
        start = time.time()
        sa=suffixArray(s)
        end = time.time()
        timings.append(end - start)
    tavg = sum(timings) / trials
    print("Average time = {:.4f} sec".format(tavg))
    if dtype == 'str':
        offsets_transferred = 0
        bytes_transferred = len(s)
        bytes_per_sec = (offsets_transferred + bytes_transferred) / tavg
    else:
        print("Wrong data type")
    print("Average rate = {:.2f} GiB/sec".format(bytes_per_sec/2**30))

def check_correctness( vsize,strlen, trials, dtype):
    Ni = strlen
    Nv = vsize

    v = ak.random_strings_uniform(1, Ni, Nv)
    c=ak.suffix_array(v)
    for k in range(Nv):
        s=v[k]
        sa=suffixArray(s)
        aksa=c[k]
#        _,tmp=c[k].split(maxsplit=1)
#        aksa=tmp.split()
#        intaksa  = [int(numeric_string) for numeric_string in aksa]
#        intaksa  = aksa[1:-1]
#        print(sa)
#        print(intaksa)
        assert (sa==aksa)


def create_parser():
    parser = argparse.ArgumentParser(description="Measure the performance of suffix array building: C= suffix_array(V)")
    parser.add_argument('hostname', help='Hostname of arkouda server')
    parser.add_argument('port', type=int, help='Port of arkouda server')
    parser.add_argument('-n', '--size', type=int, default=10**4, help='Problem size: length of strings')
    parser.add_argument('-v', '--number', type=int, default=10,help='Number of strings')
    parser.add_argument('-t', '--trials', type=int, default=6, help='Number of times to run the benchmark')
    parser.add_argument('-d', '--dtype', default='str', help='Dtype of value array ({})'.format(', '.join(TYPES)))
    parser.add_argument('--numpy', default=False, action='store_true', help='Run the same operation in NumPy to compare performance.')
    parser.add_argument('-r', '--randomize', default=False, action='store_true', help='Use random values instead of ones')
    parser.add_argument('--correctness-only', default=False, action='store_true', help='Only check correctness, not performance.')
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
        check_correctness(args.number, args.size, args.trials, args.dtype)
        print("CORRECT")
        sys.exit(0)


    print("length of strings = {:,}".format(args.size))
    print("number of strings = {:,}".format(args.number))
    print("number of trials = ", args.trials)
    time_ak_sa(args.number, args.size, args.trials, args.dtype)
    if args.numpy:
        time_np_sa(args.number, args.size, args.trials, args.dtype)
    sys.exit(0)
