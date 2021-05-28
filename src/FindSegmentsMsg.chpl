module FindSegmentsMsg
{
    use ServerConfig;
    
    use Time only;
    use Math only;
    use Reflection;
    use Errors;
    use Logging;
    use Message;
    
    use MultiTypeSymbolTable;
    use MultiTypeSymEntry;
    use ServerErrorStrings;
    use SegmentedArray;

    use PrivateDist;
    use CommAggregation;

    private config const logLevel = ServerConfig.logLevel;
    const fsLogger = new Logger(logLevel);

    /*

    :arg reqMsg: request containing (cmd,kname,pname)
    :type reqMsg: string 

    :arg st: SymTab to act on
    :type st: borrowed SymTab 

    :returns: (MsgTuple) 
    :throws: `UndefinedSymbolError(name)`

    */
    proc findSegmentsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (pname, nkeysStr, rest) = payload.splitMsgToTuple(3);
        var nkeys = nkeysStr:int; // number of key arrays
        var fields = rest.split(); // split request into fields
        if (fields.size != 2*nkeys) { 
             var errorMsg = incompatibleArgumentsError(pn, 
                       "Expected %i arrays but got %i".format(nkeys, (fields.size - 3)/2));
             fsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
             return new MsgTuple(errorMsg, MsgType.ERROR);                    
        }
        var low = fields.domain.low;
        var knames = fields[low..#nkeys]; // key arrays
        var ktypes = fields[low+nkeys..#nkeys]; // objtypes
        var size: int;
        // Check all the argument arrays before doing anything
        var gPerm = st.lookup(pname);
        if (gPerm.dtype != DType.Int64) { 
            var errorMsg = notImplementedError(pn,"(permutation dtype "+dtype2str(gPerm.dtype)+")"); 
            fsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }        
        // var keyEntries: [0..#nkeys] borrowed GenSymEntry;
        for (name, objtype, i) in zip(knames, ktypes, 1..) {
      // var g: borrowed GenSymEntry;
      var thisSize: int;
      var thisType: DType;
      select objtype {
          when "pdarray" {
              var g = st.lookup(name);
              thisSize = g.size;
              thisType = g.dtype;
          }
          when "str" {
              var (myNames,_) = name.splitMsgToTuple('+', 2);
              var g = st.lookup(myNames);
              thisSize = g.size;
              thisType = g.dtype;
          }
          otherwise {
              var errorMsg = unrecognizedTypeError(pn, objtype);
              fsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);          
          }
      }
      if (i == 1) {
        size = thisSize;
      } else {
          if (thisSize != size) { 
              var errorMsg = incompatibleArgumentsError(pn, 
                                "Expected array of size %i, got size %i".format(size, thisSize)); 
              fsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);                        
          }
      }
          if (thisType != DType.Int64) { 
              var errorMsg = notImplementedError(pn,"(key array dtype "+dtype2str(thisType)+")");
              fsLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
          }
      }
        
        // At this point, all arg arrays exist, have the same size, and are int64 or string dtype
        if (size == 0) {
            // Return two empty integer entries
            var n1 = st.nextName();
            st.addEntry(n1, 0, int);
            var n2 = st.nextName();
            st.addEntry(n2, 0, int);

            repMsg = "created " + st.attrib(n1) + " +created " + st.attrib(n1);
            fsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }

        // Permutation that groups the keys
        var perm = toSymEntry(gPerm, int);
        ref pa = perm.a; // ref to actual permutation array
        ref paD = perm.aD; // ref to domain
        // Unique key indices; true where first value of each unique key occurs
        var ukeylocs: [paD] bool;
        // First key is automatically present
        ukeylocs[0] = true;
        for (name, objtype) in zip(knames, ktypes) {
      select objtype {
        when "pdarray" {
          var g: borrowed GenSymEntry = st.lookup(name);
          var k = toSymEntry(g,int); // key array
          ref ka = k.a; // ref to key array
          // Permute the key array to grouped order
          var permKey: [paD] int;
          forall (s, p) in zip(permKey, pa) with (var agg = newSrcAggregator(int)) { 
            agg.copy(s, ka[p]);
          }
          // Find steps and update ukeylocs
          [(u, s, i) in zip(ukeylocs, permKey, paD)] if ((i > paD.low) && (permKey[i-1] != s))  { u = true; }
        }
        when "str" {
          // TODO remvove legacy_placeholder
          var (myNames1,legacy_placeholder) = name.splitMsgToTuple('+', 2);
          fsLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                              "findSegmentsMessage myNames1: {} legacy_placeholder: {}".format(myNames1,legacy_placeholder));
          var str = getSegString(myNames1, st);
          var (permOffsets, permVals) = str[pa];
          const ref D = permOffsets.domain;
          var permLengths: [D] int;
          permLengths[D.interior(-(D.size-1))] = permOffsets[D.interior(D.size-1)] - permOffsets[D.interior(-(D.size-1))];
          permLengths[D.high] = permVals.domain.high - permOffsets[D.high] + 1;
          // Find steps and update ukeylocs
          // [(u, s, i) in zip(ukeylocs, permKey, paD)] if ((i > paD.low) && (permKey[i-1] != s))  { u = true; }
          forall (u, o, l, i) in zip(ukeylocs, permOffsets, permLengths, D) {
            if (i > D.low) {
              // If string lengths don't match, mark a step
              if (permLengths[i-1] != l) {
                u = true;
              } else {
                // Have to compare bytes of previous string to current string
                // If any bytes differ, mark a step
                for pos in 0..#l {
                  if permVals[permOffsets[i-1]+pos] != permVals[o+pos] {
                    u = true;
                    break;
                  }
                }
              }
            }
          } // forall
        } // when "str"
      } // select objtype
        }
    // All keys have been processed, all steps have been found
        // +scan to compute segment position... 1-based because of inclusive-scan
        var iv: [ukeylocs.domain] int = (+ scan ukeylocs);
        // compute how many segments
        var pop = iv[iv.size-1];
        fsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                 "pop = %t last-scan = %t".format(pop,iv[iv.size-1]));
        // Create SymEntry to hold segment boundaries
        var sname = st.nextName(); // segments
        var segs = st.addEntry(sname, pop, int);
        ref sa = segs.a;
        ref saD = segs.aD;
        // segment position... 1-based needs to be converted to 0-based because of inclusive-scan
        // where ever a segment break (true value) is... that index is a segment start index
        forall i in ukeylocs.domain with (var agg = newDstAggregator(int)) {
          if (ukeylocs[i] == true) {
            var idx = i; 
            agg.copy(sa[iv[i]-1], idx);
          }
        }
        // Create SymEntry to hold indices of unique keys in original (unpermuted) key arrays
        var uname = st.nextName(); // unique key indices
        var ukeyinds = st.addEntry(uname, pop, int);
        ref uka = ukeyinds.a;
        /*
         Segment boundaries are in terms of permuted arrays, so invert the permutation to get 
         back to the original index
         */
        forall (s, u) in zip(sa, uka) with (var agg = newSrcAggregator(int)) {
          agg.copy(u, pa[s]);
        }

        // Return entry names of segments and unique key indices
        repMsg =  try! "created " + st.attrib(sname) + " +created " + st.attrib(uname);
        fsLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
}
