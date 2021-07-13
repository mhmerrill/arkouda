module SegmentedSuffixArray {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use CommAggregation;
  use SipHash;
  use SegmentedArray;
  use RadixSortLSD only radixSortLSD_ranks;
  use Reflection;
  use PrivateDist;
  use ServerConfig;
  use Time only Timer, getCurrentTime;
  use Logging;
  use Errors;

  private config const logLevel = ServerConfig.logLevel;
  const saLogger = new Logger(logLevel);

  private config param useHash = true;
  param SegmentedArrayUseHash = useHash;

  private config const in1dSortThreshold = 64;

  /**
   * Represents an array of arrays, implemented as a segmented array of integers.
   * Instances are ephemeral, not stored in the symbol table. Instead, attributes
   * of this class refer to symbol table entries that persist. This class is a
   * convenience for bundling those persistent objects and defining suffix array-relevant
   * operations.
     Here we just copy SegString, we need change more in the future to fit suffix array
   */
  class SegSuffixArray {

    /**
     * The name of the SymEntry corresponding to the pdarray containing
     * the offsets, which are start indices for each string bytearray
     */
    var offsetName: string;

    /**
     * The pdarray containing the offsets, which are the start indices of
     * the bytearrays, each of whichs corresponds to an individual string.
     */
    var offsets: borrowed SymEntry(int);

    /**
     * The name of the SymEntry corresponding to the pdarray containing
     * the string values where each value is byte array.
     */
    var valueName: string;

    /**
     * The pdaray containing the complete int array composed of integer index
     * corresponding to each string,
     */
    var values: borrowed SymEntry(int);

    /**
     * The number of strings in the segmented array
     */
    var size: int;

    /**
     * The total number of bytes in the entire segmented array including
     * the bytes corresonding to the strings as well as the nulls
     * separating the string bytes.
     */
    var nBytes: int;

    /*
     * This version of the init method is the most common and is only used
     * when the names of the segments (offsets) and values SymEntries are known.
     */
    proc init(segName: string, valName: string, st: borrowed SymTab) {
      offsetName = segName;
      // The try! is needed here because init cannot throw
      var gs = try! st.lookup(segName);
      // I want this to be borrowed, but that throws a lifetime error
      var segs = toSymEntry(gs, int): unmanaged SymEntry(int);
      offsets = segs;
      valueName = valName;

      var vs = try! st.lookup(valName);
      var vals = toSymEntry(vs, int): unmanaged SymEntry(int);
      values = vals;
      size = segs.size;
      nBytes = vals.size;
    }

    /*
     * This version of init method takes segments and values arrays as
     * inputs, generates the SymEntry objects for each and passes the
     * offset and value SymTab lookup names to the alternate init method
     */
    proc init(segments: [] int, values: [] int, st: borrowed SymTab) {
      var oName = st.nextName();
      var segEntry = new shared SymEntry(segments);
      try! st.addEntry(oName, segEntry);
      var vName = st.nextName();
      var valEntry = new shared SymEntry(values);
      try! st.addEntry(vName, valEntry);
      this.init(oName, vName, st);
    }

    proc show(n: int = 3) throws {
      if (size >= 2*n) {
        for i in 0..#n {
            saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
        for i in size-n..#n {
            saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
      } else {
        for i in 0..#size {
            saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),this[i]);
        }
      }
    }

    /* Retrieve one string from the array */
    proc this(idx: int): string throws {
      if (idx < offsets.aD.low) || (idx > offsets.aD.high) {
        throw new owned OutOfBoundsError();
      }
      // Start index of the string
      var start = offsets.a[idx];
      // Index of last (null) byte in string
      var end: int;
      if (idx == size - 1) {
        end = nBytes - 1;
      } else {
        end = offsets.a[idx+1] - 1;
      }
      // Take the slice of the bytearray and "cast" it to a chpl string
      var tmp=values.a[start..end];
      var s: string;
      var i: int;
      s="";
      for i in tmp do {
        s = s + " " + i: string;
      }
      return s;
    }

    /* Take a slice of indices from the array. The slice must be a
       Chapel range, i.e. low..high by stride, not a Python slice.
       Returns arrays for the segment offsets and bytes of the slice.*/
    proc this(const slice: range(stridable=true)) throws {
      if (slice.low < offsets.aD.low) || (slice.high > offsets.aD.high) {
          saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
          "Array is out of bounds");
          throw new owned OutOfBoundsError();
      }
      // Early return for zero-length result
      if (size == 0) || (slice.size == 0) {
        return (makeDistArray(0, int), makeDistArray(0, int));
      }
      // Start of bytearray slice
      var start = offsets.a[slice.low];
      // End of bytearray slice
      var end: int;
      if (slice.high == offsets.aD.high) {
        // if slice includes the last string, go to the end of values
        end = values.aD.high;
      } else {
        end = offsets.a[slice.high+1] - 1;
      }
      // Segment offsets of the new slice
      var newSegs = makeDistArray(slice.size, int);
      ref oa = offsets.a;
      forall (i, ns) in zip(newSegs.domain, newSegs) with (var agg = newSrcAggregator(int)) {
        agg.copy(ns, oa[slice.low + i]);
      }
      // Offsets need to be re-zeroed
      newSegs -= start;
      var newVals = makeDistArray(end - start + 1, int);
      ref va = values.a;
      forall (i, nv) in zip(newVals.domain, newVals) with (var agg = newSrcAggregator(int)) {
        agg.copy(nv, va[start + i]);
      }
      return (newSegs, newVals);
    }

    /* Gather strings by index. Returns arrays for the segment offsets
       and bytes of the gathered strings.*/
    proc this(iv: [?D] int) throws {
      // Early return for zero-length result
      if (D.size == 0) {
        //return (makeDistArray(0, int), makeDistArray(0, uint(8)));
        return (makeDistArray(0, int), makeDistArray(0, int));
      }
      // Check all indices within bounds
      var ivMin = min reduce iv;
      var ivMax = max reduce iv;
      if (ivMin < 0) || (ivMax >= offsets.size) {
          saLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "Array out of bounds");
          throw new owned OutOfBoundsError();
      }
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                              "Computing lengths and offsets");
      var t1 = getCurrentTime();
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // Gather the right and left boundaries of the indexed strings
      // NOTE: cannot compute lengths inside forall because agg.copy will
      // experience race condition with loop-private variable
      var right: [D] int, left: [D] int;
      forall (r, l, idx) in zip(right, left, iv) with (var agg = newSrcAggregator(int)) {
        if (idx == high) {
          agg.copy(r, values.size);
        } else {
          agg.copy(r, oa[idx+1]);
        }
        agg.copy(l, oa[idx]);
      }
      // Lengths of segments including null bytes
      var gatheredLengths: [D] int = right - left;
      // The returned offsets are the 0-up cumulative lengths
      var gatheredOffsets = (+ scan gatheredLengths);
      // The total number of bytes in the gathered strings
      var retBytes = gatheredOffsets[D.high];
      gatheredOffsets -= gatheredLengths;

      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                "aggregation in %i seconds".format(getCurrentTime() - t1));
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Copying values");
      if logLevel == LogLevel.DEBUG {
          t1 = getCurrentTime();
      }
      var gatheredVals = makeDistArray(retBytes, int);
      // Multi-locale requires some extra localization work that is not needed
      if CHPL_COMM != 'none' {
        // Compute the src index for each byte in gatheredVals

        /* For performance, we will do this with a scan, so first we need an array
           with the difference in index between the current and previous byte. For
           the interior of a segment, this is just one, but at the segment boundary,
           it is the difference between the src offset of the current segment ("left")
           and the src index of the last byte in the previous segment (right - 1).
        */
        var srcIdx = makeDistArray(retBytes, int);
        srcIdx = 1;
        var diffs: [D] int;
        diffs[D.low] = left[D.low]; // first offset is not affected by scan
        if (D.size > 1) {
          // This expression breaks when D.size == 1, resulting in strange behavior
          // However, this logic is only necessary when D.size > 1 anyway
          diffs[D.interior(D.size-1)] = left[D.interior(D.size-1)] - (right[D.interior(-(D.size-1))] - 1);
        }
        // Set srcIdx to diffs at segment boundaries
        forall (go, d) in zip(gatheredOffsets, diffs) with (var agg = newDstAggregator(int)) {
          agg.copy(srcIdx[go], d);
        }
        srcIdx = + scan srcIdx;
        // Now srcIdx has a dst-local copy of the source index and vals can be efficiently gathered
        ref va = values.a;
        forall (v, si) in zip(gatheredVals, srcIdx) with (var agg = newSrcAggregator(int)) {
          agg.copy(v, va[si]);
        }
      } else {
        ref va = values.a;
        // Copy string data to gathered result
        forall (go, gl, idx) in zip(gatheredOffsets, gatheredLengths, iv) {
          for pos in 0..#gl {
            gatheredVals[go+pos] = va[oa[idx]+pos];
          }
        }
      }
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "Gathered offsets and vals in %i seconds".format(
                                           getCurrentTime() -t1));
      return (gatheredOffsets, gatheredVals);
    }

    /* Logical indexing (compress) of strings. */
    proc this(iv: [?D] bool) throws {
      // Index vector must be same domain as array
      if (D != offsets.aD) {
          saLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                           "Array out of bounds");
          throw new owned OutOfBoundsError();
      }
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                                 "Computing lengths and offsets");
      var t1 = getCurrentTime();
      ref oa = offsets.a;
      const low = offsets.aD.low, high = offsets.aD.high;
      // Calculate the destination indices
      var steps = + scan iv;
      var newSize = steps[high];
      steps -= iv;
      // Early return for zero-length result
      if (newSize == 0) {
        return (makeDistArray(0, int), makeDistArray(0, int));
      }
      var segInds = makeDistArray(newSize, int);
      forall (t, dst, idx) in zip(iv, steps, D) with (var agg = newDstAggregator(int)) {
        if t {
          agg.copy(segInds[dst], idx);
        }
      }
      return this[segInds];
    }

    /* Apply a hash function to all strings. This is useful for grouping
       and set membership. The hash used is SipHash128.*/
    proc hash() throws {
      // 128-bit hash values represented as 2-tuples of uint(64)
      var hashes: [offsets.aD] 2*uint(64);
      // Early exit for zero-length result
      if (size == 0) {
        return hashes;
      }
      ref oa = offsets.a;
      ref va = values.a;
      // Compute lengths of strings
      var lengths = getLengths();
      // Hash each string
      // TO DO: test on clause with aggregator
      forall (o, l, h) in zip(oa, lengths, hashes) {
        const myRange = o..#l;
        h = sipHash128(va, myRange);
      }
      return hashes;
    }

    /* Return a permutation that groups the strings. Because hashing is used,
       this permutation will not sort the strings, but all equivalent strings
       will fall in one contiguous block. */
    proc argGroup() throws {
      var t = new Timer();
      if useHash {
        // Hash all strings
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), "Hashing strings");
        if logLevel == LogLevel.DEBUG { t.start(); }
        var hashes = this.hash();

        if logLevel == LogLevel.DEBUG {
            t.stop();
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "hashing took %t seconds\nSorting hashes".format(t.elapsed()));
            t.clear(); t.start();
        }

        // Return the permutation that sorts the hashes
        var iv = radixSortLSD_ranks(hashes);
        if logLevel == LogLevel.DEBUG { 
            t.stop(); 
            saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                            "sorting took %t seconds".format(t.elapsed()));
        }
        if logLevel == LogLevel.DEBUG {
          var sortedHashes = [i in iv] hashes[i];
          var diffs = sortedHashes[(iv.domain.low+1)..#(iv.size-1)] -
                                                 sortedHashes[(iv.domain.low)..#(iv.size-1)];
          printAry("diffs = ", diffs);
          var nonDecreasing = [(d0,d1) in diffs] ((d0 > 0) || ((d0 == 0) && (d1 >= 0)));
          saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                    "Are hashes sorted? %i".format(&& reduce nonDecreasing));
        }
        return iv;
      } else {
        var iv = argsort();
        return iv;
      }
    }

    /* Return lengths of all strings, including null terminator. */
    proc getLengths() {
      var lengths: [offsets.aD] int;
      if (size == 0) {
        return lengths;
      }
      ref oa = offsets.a;
      const low = offsets.aD.low;
      const high = offsets.aD.high;
      forall (i, o, l) in zip(offsets.aD, oa, lengths) {
        if (i == high) {
          l = values.size - o;
        } else {
          l = oa[i+1] - o;
        }
      }
      return lengths;
    }

    /* The comments above is treated as though they were ediff's comment string, which will cause sphinx errors
     * It takes me several hours without any idea and thanks  Brad help out. He added the following
     * line to solve the problem
     * dummy chpldoc description for ediff()
     */
    proc ediff():[offsets.aD] int {
      var diff: [offsets.aD] int;
      if (size < 2) {
        return diff;
      }
      ref oa = offsets.a;
      ref va = values.a;
      const high = offsets.aD.high;
      forall (i, a) in zip(offsets.aD, diff) {
        if (i < high) {
          var asc: bool;
          const left = oa[i]..oa[i+1]-1;
          if (i < high - 1) {
            const right = oa[i+1]..oa[i+2]-1;
            a = -memcmp(va, left, va, right);
          } else { // i == high - 1
            const right = oa[i+1]..values.aD.high;
            a = -memcmp(va, left, va, right);
          }
        } else { // i == high
          a = 0;
        }
      }
      return diff;
    }

    proc isSorted():bool {
      if (size < 2) {
        return true;
      }
      return (&& reduce (ediff() >= 0));
    }

    proc argsort(checkSorted:bool=false): [offsets.aD] int throws {
      const ref D = offsets.aD;
      const ref va = values.a;
      if checkSorted && isSorted() {
          saLogger.warn(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "argsort called on already sorted array");
          var ranks: [D] int = [i in D] i;
          return ranks;
      }
      var ranks = twoPhaseStringSort(this);
      return ranks;
    }

  } // class SegSuffixArray



  /* Test array of strings for membership in another array (set) of strings. Returns
     a boolean vector the same size as the first array. */
  proc in1d_Int(mainSar: SegSuffixArray, testSar: SegSuffixArray, invert=false) throws where useHash {
    var truth: [mainSar.offsets.aD] bool;
    // Early exit for zero-length result
    if (mainSar.size == 0) {
      return truth;
    }
    // Hash all suffix array for fast comparison
    var t = new Timer();
    saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),"Hashing strings");
    if logLevel == LogLevel.DEBUG { t.start(); }
    const hashes = mainSar.hash();
    if logLevel == LogLevel.DEBUG {
        t.stop();
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "%t seconds".format(t.elapsed()));
        t.clear();
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "Making associative domains for test set on each locale");
        t.start();
    }
    // On each locale, make an associative domain with the hashes of the second array
    // parSafe=false because we are adding in serial and it's faster
    var localTestHashes: [PrivateSpace] domain(2*uint(64), parSafe=false);
    coforall loc in Locales {
      on loc {
        // Local hashes of second array
        ref mySet = localTestHashes[here.id];
        mySet.requestCapacity(testSar.size);
        const testHashes = testSar.hash();
        for h in testHashes {
          mySet += h;
        }
      }
    }
    if logLevel == LogLevel.DEBUG {
      t.stop();
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "%t seconds".format(t.elapsed()));
      t.clear();
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "Testing membership");
      t.start();
    }
    [i in truth.domain] truth[i] = localTestHashes[here.id].contains(hashes[i]);
    if logLevel == LogLevel.DEBUG {
        t.stop();
        saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "%t seconds".format(t.elapsed()));
    }
    return truth;
  }


  proc in1d_Int(mainSar: SegSuffixArray, testSar: SegSuffixArray, invert=false) throws where !useHash {
    var truth: [mainSar.offsets.aD] bool;
    // Early exit for zero-length result
    if (mainSar.size == 0) {
      return truth;
    }
    if (testSar.size <= in1dSortThreshold) {
      for i in 0..#testSar.size {
        truth |= (mainSar == testSar[i]);
      }
      return truth;
    } else {
      // This is inspired by numpy in1d
      const (uoMain, uvMain, cMain, revIdx) = uniqueGroup(mainSar, returnInverse=true);
      const (uoTest, uvTest, cTest, revTest) = uniqueGroup(testSar);
      const (segs, vals) = concat(uoMain, uvMain, uoTest, uvTest);
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
           "Unique strings in first array: %t\nUnique strings in second array: %t\nConcat length: %t".format(
                                             uoMain.size, uoTest.size, segs.size));
      var st = new owned SymTab();
      const ar = new owned SegSuffixArray(segs, vals, st);
      const order = ar.argsort();
      const (sortedSegs, sortedVals) = ar[order];
      const sar = new owned SegSuffixArray(sortedSegs, sortedVals, st);
      if logLevel == LogLevel.DEBUG {
          saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                            "Sorted concatenated unique strings:");
          sar.show(10);
          stdout.flush();
      }
      const D = sortedSegs.domain;
      // First compare lengths and only check pairs whose lengths are equal (because gathering them is expensive)
      var flag: [D] bool;
      const lengths = sar.getLengths();
      const ref saro = sar.offsets.a;
      const ref sarv = sar.values.a;
      const high = D.high;
      forall (i, f, o, l) in zip(D, flag, saro, lengths) {
        if (i < high) && (l == lengths[i+1]) {
          const left = o..saro[i+1]-1;
          var eq: bool;
          if (i < high - 1) {
            const right = saro[i+1]..saro[i+2]-1;
            eq = (memcmp(sarv, left, sarv, right) == 0);
          } else {
            const ref right = saro[i+1]..sar.values.aD.high;
            eq = (memcmp(sarv, left, sarv, right) == 0);
          }
          if eq {
            f = true;
            flag[i+1] = true;
          }
        }
      }
      saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "Flag pop: %t".format(+ reduce flag));

      // Now flag contains true for both elements of duplicate pairs
      if invert {flag = !flag;}
      // Permute back to unique order
      var ret: [D] bool;
      forall (o, f) in zip(order, flag) with (var agg = newDstAggregator(bool)) {
        agg.copy(ret[o], f);
      }
      if logLevel == LogLevel.DEBUG {
          saLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "Ret pop: %t".format(+ reduce ret));
      }
      // Broadcast back to original (pre-unique) order
      var truth: [mainSar.offsets.aD] bool;
      forall (t, i) in zip(truth, revIdx) with (var agg = newSrcAggregator(bool)) {
        agg.copy(t, ret[i]);
      }
      return truth;
    }
  }
}
