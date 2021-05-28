module ConcatenateMsg
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
    use SegmentedArray;
    use ServerErrorStrings;
    use CommAggregation;
    use PrivateDist;
    
    use AryUtil;
    
    private config const logLevel = ServerConfig.logLevel;
    const cmLogger = new Logger(logLevel);

    /* Concatenate a list of arrays together
       to form one array
     */
    proc concatenateMsg(cmd: string, payload: string, st: borrowed SymTab) : MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string;
        var (nstr, objtype, mode, rest) = payload.splitMsgToTuple(4);
        var n = try! nstr:int; // number of arrays to sort
        var fields = rest.split();
        const low = fields.domain.low;
        var names = fields[low..];
        
        cmLogger.debug(getModuleName(),getRoutineName(), getLineNumber(), 
              "number of arrays: %i fields: %t low: %t names: %t".format(n,fields,low,names));

        // Check that fields contains the stated number of arrays
        if (n != names.size) { 
            var errorMsg = incompatibleArgumentsError(pn, 
                             "Expected %i arrays but got %i".format(n, names.size)); 
            cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                               
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        /* var arrays: [0..#n] borrowed GenSymEntry; */
        var size: int = 0;
        var blocksizes: [PrivateSpace] int;
        var blockValSizes: [PrivateSpace] int;
        var nbytes: int = 0;          
        var dtype: DType;
        // Check that all arrays exist in the symbol table and have the same size
        for (rawName, i) in zip(names, 1..) {
            // arrays[i] = st.lookup(name): borrowed GenSymEntry;
            var name: string;
            var valSize: int;
            select objtype {
                when "str" {
                    var valName: string;
                    (name, valName) = rawName.splitMsgToTuple('+', 2);
                    try {
                        // get the values/bytes portion of strings
                        var segString = getSegString(name, st);
                        nbytes += segString.nBytes;
                        valSize = segString.nBytes;
                    } catch e: Error {
                        throw getErrorWithContext(
                           msg="lookup for %s failed".format(name),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="UnknownSymbolError");                    
                    }
                    cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                             "name: %s valName: %s".format(name,valName));
                }
                when "pdarray" {
                    name = rawName;
                    cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                 "pdarray name %s".format(rawName));
                }
                otherwise { 
                    var errorMsg = notImplementedError(pn, objtype); 
                    cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
                    return new MsgTuple(errorMsg,MsgType.ERROR);                  
                }
            }
            var g: borrowed GenSymEntry;
            
            try { 
                g = st.lookup(name);
            } catch e : Error {
                throw getErrorWithContext(
                           msg="lookup for %s failed".format(name),
                           lineNumber=getLineNumber(),
                           routineName=getRoutineName(), 
                           moduleName=getModuleName(),
                           errorClass="UnknownSymbolError");
            }
            if (i == 1) {dtype = g.dtype;}
            else {
                if (dtype != g.dtype) {
                    var errorMsg = incompatibleArgumentsError(pn, 
                             "Expected %s dtype but got %s dtype".format(dtype2str(dtype), 
                                    dtype2str(g.dtype)));
                    cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg,MsgType.ERROR);
                }
            }
            // accumulate size from each array size
            size += g.size;
            if mode == "interleave" {
              const dummyDomain = makeDistDom(g.size);
              coforall loc in Locales {
                on loc {
                  const mynumsegs = dummyDomain.localSubdomain().size;
                  blocksizes[here.id] += mynumsegs;
                  /* If the size of the array is less than the number of locales,
                   * some locales will have no segments. For those locales, skip
                   * the byte size computation because, not only is it unnecessary,
                   * but it also causes an out of bounds array index due to the 
                   * way the low and high of the empty domain are computed. 
                   */
                  if (objtype == "str") && (mynumsegs > 0) {
                    const stringEntry = toSegStringSymEntry(g);
                    const e = stringEntry.offsetsEntry;
                    const firstSeg = e.a[e.aD.localSubdomain().low];
                    var mybytes: int;
                    /* If this locale contains the last segment, we cannot use the
                     * next segment offset to calculate the number of bytes for this
                     * locale, and we must instead use the total size of the values
                     * array.
                     */
                    if (e.aD.localSubdomain().high >= e.aD.high) {
                      mybytes = valSize - firstSeg;
                    } else {
                      mybytes = e.a[e.aD.localSubdomain().high + 1] - firstSeg;
                    }
                    blockValSizes[here.id] += mybytes;
                  }
                }
              }
            }
        }
        var blockstarts: [PrivateSpace] int;
        var blockValStarts: [PrivateSpace] int;
        if mode == "interleave" {
          blockstarts = (+ scan blocksizes) - blocksizes;
          blockValStarts = (+ scan blockValSizes) - blockValSizes;
        }

        // allocate a new array in the symboltable
        // and copy in arrays
        select objtype {
            when "str" {
                // var segName = st.nextName();
                // var esegs = st.addEntry(segName, size, int);
                // var valName = st.nextName();
                // var evals = st.addEntry(valName, nbytes, uint(8));
                // Allocate the two components of a Segmented
                var esegs = createTypedSymEntry(size, int);
                var evals = createTypedSymEntry(nbytes, uint(8));
                ref esa = esegs.a;
                ref eva = evals.a;
                var segStart = 0;
                var valStart = 0;

                // Let's allocate a new SegString for the return object
                var retString = assembleSegStringFromParts(esegs, evals, st);
                for (rawName, i) in zip(names, 1..) {
                    var (strName, legacy_placerholder) = rawName.splitMsgToTuple('+', 2);
                    var segString = getSegString(strName, st);
                    // var thisSegs = toSymEntry(st.lookup(segName), int);
                    var thisSegs = segString.offsets;
                    var newSegs = thisSegs.a + valStart;
                    // var thisVals = toSymEntry(st.lookup(valName), uint(8));
                    var thisVals = segString.values;
                    if mode == "interleave" {
                      coforall loc in Locales {
                        on loc {
                          // Number of strings on this locale for this input array
                          const mynsegs = thisSegs.aD.localSubdomain().size;
                          // If no strings on this locale, skip to avoid out of bounds array
                          // accesses
                          if mynsegs > 0 {
                            ref mysegs = thisSegs.a.localSlice[thisSegs.aD.localSubdomain()];
                            // Segments must be rebased to start from blockValStart,
                            // which is the current pointer to this locale's chunk of
                            // the values array
                            esegs.a[{blockstarts[here.id]..#mynsegs}] = mysegs - mysegs[thisSegs.aD.localSubdomain().low] + blockValStarts[here.id];
                            blockstarts[here.id] += mynsegs;
                            const firstSeg = thisSegs.a[thisSegs.aD.localSubdomain().low];
                            var mybytes: int;
                            // If locale contains last string, must use overall number of bytes
                            // to compute size, instead of start of next string
                            if (thisSegs.aD.localSubdomain().high >= thisSegs.aD.high) {
                              mybytes = thisVals.size - firstSeg;
                            } else {
                              mybytes = thisSegs.a[thisSegs.aD.localSubdomain().high + 1] - firstSeg;
                            }
                            evals.a[{blockValStarts[here.id]..#mybytes}] = thisVals.a[firstSeg..#mybytes];
                            blockValStarts[here.id] += mybytes;
                          }
                        }
                      }
                    } else {
                      forall (i, s) in zip(newSegs.domain, newSegs) with (var agg = newDstAggregator(int)) {
                        agg.copy(esa[i+segStart], s);
                      }
                      forall (i, v) in zip(thisVals.aD, thisVals.a) with (var agg = newDstAggregator(uint(8))) {
                        agg.copy(eva[i+valStart], v);
                      }
                      segStart += thisSegs.size;
                      valStart += thisVals.size;
                    }
                }
                var repMsg = "created " + st.attrib(retString.name) + "+created bytes.size %t".format(retString.nBytes);
                // var repMsg = "created " + st.attrib(retString.name) + "+created " + st.attrib(retString.name);
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                  "created concatenated pdarray %s".format(st.attrib(retString.name)));
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            when "pdarray" {
                var rname = st.nextName();
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), 
                                             "creating pdarray %s of type %t".format(rname,dtype));
                select (dtype) {
                    when DType.Int64 {
                        // create array to copy into
                        var e = st.addEntry(rname, size, int);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(st.lookup(name), int);
                            if mode == "interleave" {
                              coforall loc in Locales {
                                on loc {
                                  const size = o.aD.localSubdomain().size;
                                  e.a[{blockstarts[here.id]..#size}] = o.a.localSlice[o.aD.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = e.a;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.aD, o.a) with (var agg = newDstAggregator(int)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.Float64 {
                        // create array to copy into
                        var e = st.addEntry(rname, size, real);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(st.lookup(name), real);
                            if mode == "interleave" {
                              coforall loc in Locales {
                                on loc {
                                  const size = o.aD.localSubdomain().size;
                                  e.a[{blockstarts[here.id]..#size}] = o.a.localSlice[o.aD.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = e.a;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.aD, o.a) with (var agg = newDstAggregator(real)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    when DType.Bool {
                        // create array to copy into
                        var e = st.addEntry(rname, size, bool);
                        var start: int;
                        start = 0;
                        for (name, i) in zip(names, 1..) {
                            // lookup and cast operand to copy from
                            const o = toSymEntry(st.lookup(name), bool);
                            if mode == "interleave" {
                              coforall loc in Locales {
                                on loc {
                                  const size = o.aD.localSubdomain().size;
                                  e.a[{blockstarts[here.id]..#size}] = o.a.localSlice[o.aD.localSubdomain()];
                                  blockstarts[here.id] += size;
                                }
                              }
                            } else {
                              ref ea = e.a;
                              // copy array into concatenation array
                              forall (i, v) in zip(o.aD, o.a) with (var agg = newDstAggregator(bool)) {
                                agg.copy(ea[start+i], v);
                              }
                              // update new start for next array copy
                              start += o.size;
                            }
                        }
                        cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                           "created concatenated pdarray: %s".format(st.attrib(rname)));
                    }
                    otherwise {
                        var errorMsg = notImplementedError("concatenate",dtype);
                        cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg,MsgType.ERROR);                      
                    }
                }

                repMsg = "created " + st.attrib(rname);
                cmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                return new MsgTuple(repMsg, MsgType.NORMAL);
            }
            otherwise { 
                var errorMsg = notImplementedError(pn, objtype); 
                cmLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    }
}
