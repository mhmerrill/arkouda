module SegmentedMsg {
  use Reflection;
  use Errors;
  use Logging;
  use Message;
  use SegmentedArray;
  use ServerErrorStrings;
  use ServerConfig;
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use RandArray;
  use IO;
  use GenSymIO only jsonToPdArray,jsonToPdArrayInt;

  use SymArrayDmap;
  use SACA;


  private config const logLevel = ServerConfig.logLevel;
  const smLogger = new Logger(logLevel);

  proc randomStringsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var (lenStr, dist, charsetStr, arg1str, arg2str, seedStr)
          = payload.splitMsgToTuple(6);
      var len = lenStr: int;
      var charset = str2CharSet(charsetStr);
      var segName = st.nextName();
      var valName = st.nextName();
      var repMsg: string;
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
             "dist: %s segName: %t valName: %t".format(dist.toLower(),segName,valName));
      select dist.toLower() {
          when "uniform" {
              var minLen = arg1str:int;
              var maxLen = arg2str:int;
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + (maxLen + minLen)*len);
              var (segs, vals) = newRandStringsUniformLength(len, minLen, maxLen, charset, seedStr);
              var segEntry = new shared SymEntry(segs);
              var valEntry = new shared SymEntry(vals);
              st.addEntry(segName, segEntry);
              st.addEntry(valName, valEntry);
              repMsg = 'created ' + st.attrib(segName) + '+created ' + st.attrib(valName);
          }
          when "lognormal" {
              var logMean = arg1str:real;
              var logStd = arg2str:real;
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*len + 16*len + exp(logMean + (logStd**2)/2):int*len);
              var (segs, vals) = newRandStringsLogNormalLength(len, logMean, logStd, charset, seedStr);
              var segEntry = new shared SymEntry(segs);
              var valEntry = new shared SymEntry(vals);
              st.addEntry(segName, segEntry);
              st.addEntry(valName, valEntry);
              repMsg = 'created ' + st.attrib(segName) + '+created ' + st.attrib(valName);
          }
          otherwise { 
              var errorMsg = notImplementedError(pn, dist);      
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);  
              return new MsgTuple(errorMsg, MsgType.ERROR);    
          }
      }

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);      
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentLengthsMsg(cmd: string, payload: string, 
                                          st: borrowed SymTab): MsgTuple throws {


    var pn = Reflection.getRoutineName();
    var (objtype, segName, valName) = payload.splitMsgToTuple(3);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);
    
    var rname = st.nextName();
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
            "cmd: %s objtype: %t segName: %t valName: %t".format(
                   cmd,objtype,segName,valName));

    select objtype {
      when "str" {
        var strings = getSegString(segName, valName, st);
        var lengths = st.addEntry(rname, strings.size, int);
        // Do not include the null terminator in the length
        lengths.a = strings.getLengths() - 1;
      }
      when "int" {
        var sarrays = new owned SegSArray(segName, valName, st);
        var lengths = st.addEntry(rname, sarrays.size, int);
        // Do not include the null terminator in the length
        lengths.a = sarrays.getLengths() - 1;
      }
      otherwise {
          var errorMsg = notImplementedError(pn, "%s".format(objtype));
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                      
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }

    var repMsg = "created "+st.attrib(rname);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedEfuncMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (subcmd, objtype, segName, valName, valtype, valStr) = 
                                              payload.splitMsgToTuple(6);

      // check to make sure symbols defined
      st.checkTable(segName);
      st.checkTable(valName);

      var json = jsonToPdArray(valStr, 1);
      var val = json[json.domain.low];
      var rname = st.nextName();
    
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));
    
        select (objtype, valtype) {
          when ("str", "str") {
            var strings = getSegString(segName, valName, st);
            select subcmd {
                when "contains" {
                var truth = st.addEntry(rname, strings.size, bool);
                truth.a = strings.substringSearch(val, SearchMode.contains);
                repMsg = "created "+st.attrib(rname);
            }
            when "startswith" {
                var truth = st.addEntry(rname, strings.size, bool);
                truth.a = strings.substringSearch(val, SearchMode.startsWith);
                repMsg = "created "+st.attrib(rname);
            }
            when "endswith" {
                var truth = st.addEntry(rname, strings.size, bool);
                truth.a = strings.substringSearch(val, SearchMode.endsWith);
                repMsg = "created "+st.attrib(rname);
            }
            otherwise {
               var errorMsg = notImplementedError(pn, "subcmd: %s, (%s, %s)".format(
                         subcmd, objtype, valtype));
               smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
               return new MsgTuple(errorMsg, MsgType.ERROR);
            }
          }
        }
        otherwise {
          var errorMsg = "(%s, %s)".format(objtype, valtype);
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(notImplementedError(pn, errorMsg), MsgType.ERROR);
        }
      }

      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

proc segmentedPeelMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (subcmd, objtype, segName, valName, valtype, valStr,
         idStr, kpStr, lStr, jsonStr) = payload.splitMsgToTuple(10);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);

    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                         "cmd: %s subcmd: %s objtype: %t valtype: %t".format(
                          cmd,subcmd,objtype,valtype));

    select (objtype, valtype) {
    when ("str", "str") {
      var strings = getSegString(segName, valName, st);
      select subcmd {
        when "peel" {
          var times = valStr:int;
          var includeDelimiter = (idStr.toLower() == "true");
          var keepPartial = (kpStr.toLower() == "true");
          var left = (lStr.toLower() == "true");
          var json = jsonToPdArray(jsonStr, 1);
          var val = json[json.domain.low];
          var loname = st.nextName();
          var lvname = st.nextName();
          var roname = st.nextName();
          var rvname = st.nextName();
          select (includeDelimiter, keepPartial, left) {
          when (false, false, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, false, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (false, false, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, false, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (false, true, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, true, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (false, true, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, false, true, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, false, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, false, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, false, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, false, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, true, false) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, true, false);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } when (true, true, true) {
            var (lo, lv, ro, rv) = strings.peel(val, times, true, true, true);
            st.addEntry(loname, new shared SymEntry(lo));
            st.addEntry(lvname, new shared SymEntry(lv));
            st.addEntry(roname, new shared SymEntry(ro));
            st.addEntry(rvname, new shared SymEntry(rv));
          } otherwise {
              var errorMsg = notImplementedError(pn, 
                               "subcmd: %s, (%s, %s)".format(subcmd, objtype, valtype));
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);                            
              }
          }
          repMsg = "created %s+created %s+created %s+created %s".format(st.attrib(loname),
                                                                        st.attrib(lvname),
                                                                        st.attrib(roname),
                                                                        st.attrib(rvname));
        }
        otherwise {
            var errorMsg = notImplementedError(pn, 
                              "subcmd: %s, (%s, %s)".format(subcmd, objtype, valtype));
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);                                          
        }
      }
    }
    otherwise {
        var errorMsg = notImplementedError(pn, "(%s, %s)".format(objtype, valtype));
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);       
      }
    }
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segmentedHashMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (objtype, segName, valName) = payload.splitMsgToTuple(3);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);

    select objtype {
        when "str" {
            var strings = getSegString(segName, valName, st);
            var hashes = strings.hash();
            var name1 = st.nextName();
            var hash1 = st.addEntry(name1, hashes.size, int);
            var name2 = st.nextName();
            var hash2 = st.addEntry(name2, hashes.size, int);
            forall (h, h1, h2) in zip(hashes, hash1.a, hash2.a) {
                (h1,h2) = h:(int,int);
            }
            var repMsg = "created " + st.attrib(name1) + "+created " + st.attrib(name2);
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
/*
        when "int" {
            var sarrays = new owned SegSArray(segName, valName, st);
            var hashes = sarrays.hash();
            var name1 = st.nextName();
            var hash1 = st.addEntry(name1, hashes.size, int);
            var name2 = st.nextName();
            var hash2 = st.addEntry(name2, hashes.size, int);
            forall (h, h1, h2) in zip(hashes, hash1.a, hash2.a) {
                (h1,h2) = h:(int,int);
            }
            return "created " + st.attrib(name1) + "+created " + st.attrib(name2);
        }
*/
        otherwise {
            var errorMsg = notImplementedError(pn, objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
    }
  }


  /*
   * Assigns a segIntIndex, sliceIndex, or pdarrayIndex to the incoming payload
   * consisting of a sub-command, object type, offset SymTab key, array SymTab
   * key, and index value for the incoming payload.
   * 
   * Note: the sub-command indicates the index type which can be one of the following:
   * 1. intIndex : setIntIndex
   * 2. sliceIndex : segSliceIndex
   * 3. pdarrayIndex : segPdarrayIndex
  */ 
  proc segmentedIndexMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    // 'subcmd' is the type of indexing to perform
    // 'objtype' is the type of segmented array
    var (subcmd, objtype, rest) = payload.splitMsgToTuple(3);
    var fields = rest.split();
    var args: [1..#fields.size] string = fields; // parsed by subroutines
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                            "subcmd: %s objtype: %s rest: %s".format(subcmd,objtype,rest));
    try {
        select subcmd {
            when "intIndex" {
                return segIntIndex(objtype, args, st);
            }
            when "sliceIndex" {
                return segSliceIndex(objtype, args, st);
            }
            when "pdarrayIndex" {
                return segPdarrayIndex(objtype, args, st);
            }
            otherwise {
                var errorMsg = "Error in %s, unknown subcommand %s".format(pn, subcmd);
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
    } catch e: OutOfBoundsError {
        var errorMsg = "index out of bounds";
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);
    } catch e: Error {
        var errorMsg = "unknown cause %t".format(e);
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);
    }
  }
 
  /*
  Returns the object corresponding to the index
  */ 
  proc segIntIndex(objtype: string, args: [] string, 
                                         st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();

      // check to make sure symbols defined
      st.checkTable(args[1]);
      st.checkTable(args[2]);
      
      select objtype {
          when "str" {
              // Make a temporary strings array
              var strings = getSegString(args[1], args[2], st);
              // Parse the index
              var idx = args[3]:int;
              // TO DO: in the future, we will force the client to handle this
              idx = convertPythonIndexToChapel(idx, strings.size);
              var s = strings[idx];

              var repMsg = "item %s %jt".format("str", s);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
              return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when "int" {
              // Make a temporary int array
              var arrays = new owned SegSArray(args[1], args[2], st);
              // Parse the index
              var idx = args[3]:int;
              // TO DO: in the future, we will force the client to handle this
              idx = convertPythonIndexToChapel(idx, arrays.size);
              var s = arrays[idx];
              var repMsg="item %s %jt".format("int", s);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
              return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          otherwise { 
              var errorMsg = notImplementedError(pn, objtype); 
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);                          
          }
      }
  }

  /* Allow Python-style negative indices. */
  proc convertPythonIndexToChapel(pyidx: int, high: int): int {
    var chplIdx: int;
    if (pyidx < 0) {
      chplIdx = high + 1 + pyidx;
    } else {
      chplIdx = pyidx;
    }
    return chplIdx;
  }

  proc segSliceIndex(objtype: string, args: [] string, 
                                         st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();

    // check to make sure symbols defined
    st.checkTable(args[1]);
    st.checkTable(args[2]);

    select objtype {
        when "str" {
            // Make a temporary string array
            var strings = getSegString(args[1], args[2], st);

            // Parse the slice parameters
            var start = args[3]:int;
            var stop = args[4]:int;
            var stride = args[5]:int;

            // Only stride-1 slices are allowed for now
            if (stride != 1) { 
                var errorMsg = notImplementedError(pn, "stride != 1"); 
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            // TO DO: in the future, we will force the client to handle this
            var slice: range(stridable=true) = convertPythonSliceToChapel(start, stop, stride);
            var newSegName = st.nextName();
            var newValName = st.nextName();
            // Compute the slice
            var (newSegs, newVals) = strings[slice];
            // Store the resulting offsets and bytes arrays
            var newSegsEntry = new shared SymEntry(newSegs);
            var newValsEntry = new shared SymEntry(newVals);
            st.addEntry(newSegName, newSegsEntry);
            st.addEntry(newValName, newValsEntry);
            var repMsg = "created " + st.attrib(newSegName) + " +created " + st.attrib(newValName);
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
            return new MsgTuple(repMsg, MsgType.NORMAL);
      }
      when "int" {
            // Make a temporary integer  array
            var sarrays = new owned SegSArray(args[1], args[2], st);
            // Parse the slice parameters
            var start = args[3]:int;
            var stop = args[4]:int;
            var stride = args[5]:int;
            // Only stride-1 slices are allowed for now
            if (stride != 1) { 
                var errorMsg = notImplementedError(pn, "stride != 1"); 
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
            // TO DO: in the future, we will force the client to handle this
            var slice: range(stridable=true) = convertPythonSliceToChapel(start, stop, stride);
            var newSegName = st.nextName();
            var newValName = st.nextName();
            // Compute the slice
            var (newSegs, newVals) = sarrays[slice];
            // Store the resulting offsets and bytes arrays
            var newSegsEntry = new shared SymEntry(newSegs);
            var newValsEntry = new shared SymEntry(newVals);
            st.addEntry(newSegName, newSegsEntry);
            st.addEntry(newValName, newValsEntry);
        
            var repMsg = "created " + st.attrib(newSegName) + " +created " + st.attrib(newValName);
            smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg); 
            return new MsgTuple(repMsg, MsgType.NORMAL);
        }
        otherwise {
            var errorMsg = notImplementedError(pn, objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
            return new MsgTuple(errorMsg, MsgType.ERROR);          
        }
    }
  }

  proc convertPythonSliceToChapel(start:int, stop:int, stride:int=1): range(stridable=true) {
    var slice: range(stridable=true);
    // convert python slice to chapel slice
    // backwards iteration with negative stride
    if  (start > stop) & (stride < 0) {slice = (stop+1)..start by stride;}
    // forward iteration with positive stride
    else if (start <= stop) & (stride > 0) {slice = start..(stop-1) by stride;}
    // BAD FORM start < stop and stride is negative
    else {slice = 1..0;}
    return slice;
  }

  proc segPdarrayIndex(objtype: string, args: [] string, 
                                 st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();

    // check to make sure symbols defined
    st.checkTable(args[1]);
    st.checkTable(args[2]);

    var newSegName = st.nextName();
    var newValName = st.nextName();
    
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                  "objtype:%s".format(objtype));
    
    select objtype {
        when "str" {
            var strings = getSegString(args[1], args[2], st);
            var iname = args[3];
            var gIV: borrowed GenSymEntry = st.lookup(iname);
            try {
                select gIV.dtype {
                    when DType.Int64 {
                        var iv = toSymEntry(gIV, int);
                        var (newSegs, newVals) = strings[iv.a];
                        var newSegsEntry = new shared SymEntry(newSegs);
                        var newValsEntry = new shared SymEntry(newVals);
                        st.addEntry(newSegName, newSegsEntry);
                        st.addEntry(newValName, newValsEntry);
                    }
                    when DType.Bool {
                        var iv = toSymEntry(gIV, bool);
                        var (newSegs, newVals) = strings[iv.a];
                        var newSegsEntry = new shared SymEntry(newSegs);
                        var newValsEntry = new shared SymEntry(newVals);
                        st.addEntry(newSegName, newSegsEntry);
                        st.addEntry(newValName, newValsEntry);
                    }
                    otherwise {
                        var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                        smLogger.error(getModuleName(),getRoutineName(),
                                                      getLineNumber(),errorMsg); 
                        return new MsgTuple(notImplementedError(pn,errorMsg), MsgType.ERROR);
                    }
                }
            } catch e: Error {
                var errorMsg =  e.message();
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
            }
        }
        when "int" {
            var sarrays = new owned SegSArray(args[1], args[2], st);
            var iname = args[3];
            var gIV: borrowed GenSymEntry = st.lookup(iname);
            try {
                select gIV.dtype {
                    when DType.Int64 {
                        var iv = toSymEntry(gIV, int);
                        var (newSegs, newVals) = sarrays[iv.a];
                        var newSegsEntry = new shared SymEntry(newSegs);
                        var newValsEntry = new shared SymEntry(newVals);
                        st.addEntry(newSegName, newSegsEntry);
                        st.addEntry(newValName, newValsEntry);
                    }
                    when DType.Bool {
                        var iv = toSymEntry(gIV, bool);
                        var (newSegs, newVals) = sarrays[iv.a];
                        var newSegsEntry = new shared SymEntry(newSegs);
                        var newValsEntry = new shared SymEntry(newVals);
                        st.addEntry(newSegName, newSegsEntry);
                        st.addEntry(newValName, newValsEntry);
                    }
                    otherwise {
                        var errorMsg = "("+objtype+","+dtype2str(gIV.dtype)+")";
                        smLogger.error(getModuleName(),getRoutineName(),
                                                      getLineNumber(),errorMsg); 
                        return new MsgTuple(errorMsg, MsgType.ERROR);          
                    }
                }
            } catch e: Error {
                var errorMsg= "Error: %t".format(e.message());
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                      e.message());
                return new MsgTuple(errorMsg, MsgType.ERROR);          
            }
        }
        otherwise {
            var errorMsg = "unsupported objtype: %t".format(objtype);
            smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
            return new MsgTuple(notImplementedError(pn, objtype), MsgType.ERROR);
        }
    }
    var repMsg = "created " + st.attrib(newSegName) + "+created " + st.attrib(newValName);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segBinopvvMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op,
         // Type and attrib names of left segmented array
         ltype, lsegName, lvalName,
         // Type and attrib names of right segmented array
         rtype, rsegName, rvalName, leftStr, jsonStr)
           = payload.splitMsgToTuple(9);

    // check to make sure symbols defined
    st.checkTable(lsegName);
    st.checkTable(lvalName);
    st.checkTable(rsegName);
    st.checkTable(rvalName);

    select (ltype, rtype) {
        when ("str", "str") {
            var lstrings = getSegString(lsegName, lvalName, st);
            var rstrings = getSegString(rsegName, rvalName, st);

            select op {
                when "==" {
                    var rname = st.nextName();
                    var e = st.addEntry(rname, lstrings.size, bool);
                    e.a = (lstrings == rstrings);
                    repMsg = "created " + st.attrib(rname);
                }
                when "!=" {
                    var rname = st.nextName();
                    var e = st.addEntry(rname, lstrings.size, bool);
                    e.a = (lstrings != rstrings);
                    repMsg = "created " + st.attrib(rname);
                }
                when "stick" {
                    var left = (leftStr.toLower() != "false");
                    var json = jsonToPdArray(jsonStr, 1);
                    const delim = json[json.domain.low];
                    var oname = st.nextName();
                    var vname = st.nextName();
                    if left {
                        var (newOffsets, newVals) = lstrings.stick(rstrings, delim, false);
                        st.addEntry(oname, new shared SymEntry(newOffsets));
                        st.addEntry(vname, new shared SymEntry(newVals));
                    } else {
                        var (newOffsets, newVals) = lstrings.stick(rstrings, delim, true);
                        st.addEntry(oname, new shared SymEntry(newOffsets));
                        st.addEntry(vname, new shared SymEntry(newVals));
                    }
                    repMsg = "created %s+created %s".format(st.attrib(oname), st.attrib(vname));
                    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
                }
                otherwise {
                    var errorMsg = notImplementedError(pn, ltype, op, rtype);
                    smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
              }
           }
       otherwise {
           var errorMsg = unrecognizedTypeError(pn, "("+ltype+", "+rtype+")");
           smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
           return new MsgTuple(errorMsg, MsgType.ERROR);
       } 
    }

    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segBinopvvIntMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op,
         // Type and attrib names of left segmented array
         ltype, lsegName, lvalName,
         // Type and attrib names of right segmented array
         rtype, rsegName, rvalName, leftStr, jsonStr)
           = payload.splitMsgToTuple(9);

    // check to make sure symbols defined
    st.checkTable(lsegName);
    st.checkTable(lvalName);
    st.checkTable(rsegName);
    st.checkTable(rvalName);

    select (ltype, rtype) {
        when ("int", "int") {
          var lsa = new owned SegSArray(lsegName, lvalName, st);
          var rsa = new owned SegSArray(rsegName, rvalName, st);
          select op {
              when "==" {
                var rname = st.nextName();
                var e = st.addEntry(rname, lsa.size, bool);
                e.a = (lsa == rsa);
                repMsg = "created " + st.attrib(rname);
              }
              when "!=" {
                var rname = st.nextName();
                var e = st.addEntry(rname, lsa.size, bool);
                e.a = (lsa != rsa);
                repMsg = "created " + st.attrib(rname);
              }
              otherwise {
                var errorMsg= notImplementedError(pn, ltype, op, rtype);
                smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);
              }
          }
        }
        otherwise {
          var errorMsg= unrecognizedTypeError(pn, "("+ltype+", "+rtype+")");
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        } 
    }
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }
  proc segBinopvsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (op, objtype, segName, valName, valtype, encodedVal)
          = payload.splitMsgToTuple(6);

      // check to make sure symbols defined
      st.checkTable(segName);
      st.checkTable(valName);

      var json = jsonToPdArray(encodedVal, 1);
      var value = json[json.domain.low];
      var rname = st.nextName();

      select (objtype, valtype) {
          when ("str", "str") {
              var strings = getSegString(segName, valName, st);
              select op {
                  when "==" {
                      var e = st.addEntry(rname, strings.size, bool);
                      e.a = (strings == value);
                  }
                  when "!=" {
                      var e = st.addEntry(rname, strings.size, bool);
                      e.a = (strings != value);
                  }
                  otherwise {
                      var errorMsg = notImplementedError(pn, objtype, op, valtype);
                      smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                      return new MsgTuple(errorMsg, MsgType.ERROR);
                  }
              }
          }
          otherwise {
              var errorMsg = unrecognizedTypeError(pn, "("+objtype+", "+valtype+")");
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
              return new MsgTuple(errorMsg, MsgType.ERROR);
          } 
      }

      repMsg = "created %s".format(st.attrib(rname));
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }


  proc segBinopvsIntMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (op, objtype, segName, valName, valtype, encodedVal)
          = payload.splitMsgToTuple(6);

    // check to make sure symbols defined
    st.checkTable(segName);
    st.checkTable(valName);
    var json = jsonToPdArrayInt(encodedVal, 1);
    var value = json[json.domain.low];
    var rname = st.nextName();
    select (objtype, valtype) {
    when ("int", "int") {
      var sarrays  = new owned SegSArray(segName, valName, st);
      select op {
        when "==" {
          var e = st.addEntry(rname, sarrays.size, bool);
          var tmp=sarrays[sarrays.offsets.aD.low]:int;
          e.a = (tmp == value);
        }
        when "!=" {
          var e = st.addEntry(rname, sarrays.size, bool);
          var tmp=sarrays[sarrays.offsets.aD.low]:int;
          e.a = (tmp != value);
        }
        otherwise {
          var errorMsg= notImplementedError(pn, objtype, op, valtype);
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
        }
      }
    }
    otherwise {
        var errorMsg= unrecognizedTypeError(pn, "("+objtype+", "+valtype+")");
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
        return new MsgTuple(errorMsg, MsgType.ERROR);

    } 
    }
    repMsg= "created " + st.attrib(rname);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(), repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);

  }

  proc segIn1dMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var repMsg: string;
      var (mainObjtype, mainSegName, mainValName, testObjtype, testSegName,
         testValName, invertStr) = payload.splitMsgToTuple(7);

      // check to make sure symbols defined
      st.checkTable(mainSegName);
      st.checkTable(mainValName);
      st.checkTable(testSegName);
      st.checkTable(testValName);

      var invert: bool;
      if invertStr == "True" {invert = true;
      } else if invertStr == "False" {invert = false;
      } else {
          var errorMsg = "Invalid argument in %s: %s (expected True or False)".format(pn, invertStr);
          smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
          return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    
      var rname = st.nextName();
 
      select (mainObjtype, testObjtype) {
          when ("str", "str") {
              var mainStr = getSegString(mainSegName, mainValName, st);
              var testStr = getSegString(testSegName, testValName, st);
              var e = st.addEntry(rname, mainStr.size, bool);
              if invert {
                  e.a = !in1d(mainStr, testStr);
              } else {
                  e.a = in1d(mainStr, testStr);
              }
          }
          otherwise {
              var errorMsg = unrecognizedTypeError(pn, "("+mainObjtype+", "+testObjtype+")");
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);            
          }
      }

      repMsg = "created " + st.attrib(rname);
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  // this function is added for suffix array
  proc segIn1dIntMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var pn = Reflection.getRoutineName();
    var repMsg: string;
    var (mainObjtype, mainSegName, mainValName, testObjtype, testSegName,
         testValName, invertStr) = payload.splitMsgToTuple(7);

    // check to make sure symbols defined
    st.checkTable(mainSegName);
    st.checkTable(mainValName);
    st.checkTable(testSegName);
    st.checkTable(testValName);

    var invert: bool;
    if invertStr == "True" {invert = true;}
    else if invertStr == "False" {invert = false;}
    else {

          var errorMsg="Error: Invalid argument in %s: %s (expected True or False)".format(pn, invertStr);
          return new MsgTuple(errorMsg, MsgType.ERROR);
    }
    var rname = st.nextName();
    select (mainObjtype, testObjtype) {
    when ("int", "int") {
      var mainSA = new owned SegSArray(mainSegName, mainValName, st);
      var testSA = new owned SegSArray(testSegName, testValName, st);
      var e = st.addEntry(rname, mainSA.size, bool);
      if invert {
        e.a = !in1d_Int(mainSA, testSA);
      } else {
        e.a = in1d_Int(mainSA, testSA);
      }
    }
    otherwise {
        var errorMsg = unrecognizedTypeError(pn, "("+mainObjtype+", "+testObjtype+")");
        smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
        return new MsgTuple(errorMsg, MsgType.ERROR);
      }
    }
    repMsg= "created " + st.attrib(rname);
    smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }

  proc segGroupMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var (objtype, segName, valName) = payload.splitMsgToTuple(3);

      // check to make sure symbols defined
      st.checkTable(segName);
      st.checkTable(valName);
      
      var rname = st.nextName();
      select (objtype) {
          when "str" {
              var strings = getSegString(segName, valName, st);
              var iv = st.addEntry(rname, strings.size, int);
              iv.a = strings.argGroup();
          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype+")");
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);            
          }
      }

      var repMsg =  "created " + st.attrib(rname);
      smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
      return new MsgTuple(repMsg, MsgType.NORMAL);
  }



  proc segSuffixArrayMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var (objtype, segName, valName) = payload.splitMsgToTuple(3);
      var repMsg: string;

      // check to make sure symbols defined
      st.checkTable(segName);
      st.checkTable(valName);

      //var strings = new owned SegString(segName, valName, st);
      var strings = getSegString(segName, valName, st);
      var size=strings.size;
      var nBytes = strings.nBytes;
      var length=strings.getLengths();
      var offsegs = (+ scan length) - length;
      var algorithmNum=2:int; //2:"divsufsort";1:SuffixArraySkew
      select (objtype) {
          when "str" {
              // To be checked, I am not sure if this formula can estimate the total memory requirement
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*size + 16*size + nBytes);

              //allocate an offset array
              var sasoff = offsegs;
              //allocate an values array
              var sasval:[0..(nBytes-1)] int;
              //              var lcpval:[0..(nBytes-1)] int; now we will not build the LCP array at the same time

              var i:int;
              var j:int;
              forall i in 0..(size-1) do {
              // the start position of ith string in value array

                var startposition:int;
                var endposition:int;
                startposition = offsegs[i];
                endposition = startposition+length[i]-1;
                // what we do in the select structure is filling the sasval array with correct index
                select (algorithmNum) {
                    when 1 {
                       var sasize=length[i]:int;
                       ref strArray=strings.values.a[startposition..endposition];
                       var tmparray:[0..sasize+2] int;
                       var intstrArray:[0..sasize+2] int;
                       var x:int;
                       var y:int;
                       forall (x,y) in zip ( intstrArray[0..sasize-1],
                                strings.values.a[startposition..endposition]) do x=y;
                       intstrArray[sasize]=0;
                       intstrArray[sasize+1]=0;
                       intstrArray[sasize+2]=0;
                       SuffixArraySkew(intstrArray,tmparray,sasize,256);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[0..sasize-1]) do
                               x = y;
                    }
                    when 2 {
                       var sasize=length[i]:int(32);
                       var localstrArray:[0..endposition-startposition] uint(8);
                       var a:int(8);
                       var b:int(8);
                       ref strArray=strings.values.a[startposition..endposition];
                       localstrArray=strArray;
                       //for all (a,b) in zip (localstrArray[0..sasize-1],strArray) do a=b;
                       var tmparray:[1..sasize] int(32);
                       divsufsort(localstrArray,tmparray,sasize);
                       //divsufsort(strArray,tmparray,sasize);
                       var x:int;
                       var y:int(32);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[1..sasize]) do
                            x = y;
                    }
                }

/*
// Here we calculate the lcp(Longest Common Prefix) array value
                forall j in startposition+1..endposition do{
                        var tmpcount=0:int;
                        var tmpbefore=sasval[j-1]:int;
                        var tmpcur=sasval[j]:int;
                        var tmplen=min(sasize-tmpcur, sasize-tmpbefore);
                        var tmpi:int;
                        for tmpi in 0..tmplen-1 do {
                            if (intstrArray[tmpbefore]!=intstrArray[tmpcur]) {
                                 break;
                            }                        
                            tmpcount+=1;
                        } 
                        lcpval[j]=tmpcount;
                }
*/
              }
              var segName2 = st.nextName();
              var valName2 = st.nextName();
              //              var lcpvalName = st.nextName();

              var segEntry = new shared SymEntry(sasoff);
              var valEntry = new shared SymEntry(sasval);
              //              var lcpvalEntry = new shared SymEntry(lcpval);
              /*
              valEntry.enhancedInfo=lcpvalName;
              lcpvalEntry.enhancedInfo=valName2;
              we have removed enchancedInfo.
              */
              st.addEntry(segName2, segEntry);
              st.addEntry(valName2, valEntry);
//              st.addEntry(lcpvalName, lcpvalEntry);
              repMsg = 'created ' + st.attrib(segName2) + '+created ' + st.attrib(valName2);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
              return new MsgTuple(repMsg, MsgType.NORMAL);

          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype+")");
              writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError")); 
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);            
          }
      }

  }

  proc segLCPMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
      var (objtype, segName1, valName1,segName2,valName2) = payload.splitMsgToTuple(5);
      var repMsg: string;

      // check to make sure symbols defined
      st.checkTable(segName1);
      st.checkTable(valName1);
      st.checkTable(segName2);
      st.checkTable(valName2);

      var suffixarrays = new owned SegSArray(segName1, valName1, st);
      var size=suffixarrays.size;
      var nBytes = suffixarrays.nBytes;
      var length=suffixarrays.getLengths();
      var offsegs = (+ scan length) - length;


      //var strings = new owned SegString(segName2, valName2, st);
      var strings = getSegString(segName2, valName2, st);

      select (objtype) {
          when "int" {
              // To be checked, I am not sure if this formula can estimate the total memory requirement
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*size + 16*size + nBytes);

              //allocate an offset array
              var sasoff = offsegs;
              //allocate an values array
              var lcpval:[0..(nBytes-1)] int;

              var i:int;
              var j:int;
              forall i in 0..(size-1) do {
              // the start position of ith surrix array  in value array
                var startposition:int;
                var endposition:int;
                startposition = offsegs[i];
                endposition = startposition+length[i]-1;

                var sasize=length[i]:int;
                ref sufArray=suffixarrays.values.a[startposition..endposition];
                ref strArray=strings.values.a[startposition..endposition];
// Here we calculate the lcp(Longest Common Prefix) array value
                forall j in startposition+1..endposition do{
                        var tmpcount=0:int;
                        var tmpbefore=sufArray[j-1]:int;
                        var tmpcur=sufArray[j]:int;
                        var tmplen=min(sasize-tmpcur, sasize-tmpbefore);
                        var tmpi:int;
                        for tmpi in 0..tmplen-1 do {
                            if (strArray[tmpbefore]!=strArray[tmpcur]) {
                                 break;
                            }                        
                            tmpbefore+=1;
                            tmpcur+=1;
                            tmpcount+=1;
                        } 
                        lcpval[j]=tmpcount;
                }
              }
              var lcpsegName = st.nextName();
              var lcpvalName = st.nextName();

              var lcpsegEntry = new shared SymEntry(sasoff);
              var lcpvalEntry = new shared SymEntry(lcpval);
              /*
              valEntry.enhancedInfo=lcpvalName;
              lcpvalEntry.enhancedInfo=valName2;
              we have removed enchancedInfo.
              */
              st.addEntry(lcpsegName, lcpsegEntry);
              st.addEntry(lcpvalName, lcpvalEntry);
              repMsg = 'created ' + st.attrib(lcpsegName) + '+created ' + st.attrib(lcpvalName);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
              return new MsgTuple(repMsg, MsgType.NORMAL);


          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+objtype+")");
              writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError")); 
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);            
          }
      }

  }

// directly read a string from given file and generate its suffix array
  proc segSAFileMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
      var pn = Reflection.getRoutineName();
//      var (FileName) = payload.decode().splitMsgToTuple(1);
      var FileName = payload;
      var repMsg: string;

//      var filesize:int(32);
      var filesize:int;
      var f = open(FileName, iomode.r);
      var size=1:int;
      var nBytes = f.size;
      var length:[0..0] int  =nBytes;
      var offsegs:[0..0] int =0 ;

      var sasize=nBytes:int;
      var startposition:int;
      var endposition:int;
      startposition = 0;
      endposition = nBytes-1;
      var strArray:[startposition..endposition]uint(8);
      var r = f.reader(kind=ionative);
      r.read(strArray);
      r.close();

      var segName = st.nextName();
      var valName = st.nextName();

      var segEntry = new shared SymEntry(offsegs);
      var valEntry = new shared SymEntry(strArray);
      st.addEntry(segName, segEntry);
      st.addEntry(valName, valEntry);

      var algorithmNum=2:int; //2:"divsufsort";1:SuffixArraySkew

      select ("str") {
          when "str" {
              // To be checked, I am not sure if this formula can estimate the total memory requirement
              // Lengths + 2*segs + 2*vals (copied to SymTab)
              overMemLimit(8*size + 16*size + nBytes);

              //allocate an offset array
              var sasoff = offsegs;
              //allocate a suffix array  values array and lcp array
              var sasval:[0..(nBytes-1)] int;
//              var lcpval:[0..(nBytes-1)] int;

              var i:int;
              forall i in 0..(size-1) do {
              // the start position of ith string in value array
                select (algorithmNum) {
                    when 1 {
                       var sasize=length[i]:int;
                       var tmparray:[0..sasize+2] int;
                       var intstrArray:[0..sasize+2] int;
                       var x:int;
                       var y:int;
                       forall (x,y) in zip ( intstrArray[0..sasize-1],strArray[startposition..endposition]) do x=y;
                       intstrArray[sasize]=0;
                       intstrArray[sasize+1]=0;
                       intstrArray[sasize+2]=0;
                       SuffixArraySkew(intstrArray,tmparray,sasize,256);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[0..sasize-1]) do
                               x = y;
                    }
                    when 2 {
                       var sasize=length[i]:int(32);
                       //ref strArray=strings.values.a[startposition..endposition];
                       var tmparray:[1..sasize] int(32);
                       divsufsort(strArray,tmparray,sasize);
                       var x:int;
                       var y:int(32);
                       for (x, y) in zip(sasval[startposition..endposition], tmparray[1..sasize]) do
                            x = y;
                    }
                }// end of select 
              } // end of forall
              var segName2 = st.nextName();
              var valName2 = st.nextName();
//              var lcpvalName = st.nextName();

              var segEntry = new shared SymEntry(sasoff);
              var valEntry = new shared SymEntry(sasval);
//              var lcpvalEntry = new shared SymEntry(lcpval);
              /*
              valEntry.enhancedInfo=lcpvalName;
              lcpvalEntry.enhancedInfo=valName2;
              We have removed enhancedInfo.
              */
              st.addEntry(segName2, segEntry);
              st.addEntry(valName2, valEntry);
//              st.addEntry(lcpvalName, lcpvalEntry);
              repMsg = 'created ' + st.attrib(segName2) + '+created ' + st.attrib(valName2) 
                        + '+created ' + st.attrib(segName) + '+created ' + st.attrib(valName);
              smLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
              return new MsgTuple(repMsg, MsgType.NORMAL);

          }
          otherwise {
              var errorMsg = notImplementedError(pn, "("+FileName+")");
              writeln(generateErrorContext(
                                     msg=errorMsg, 
                                     lineNumber=getLineNumber(), 
                                     moduleName=getModuleName(), 
                                     routineName=getRoutineName(), 
                                     errorClass="NotImplementedError")); 
              smLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);      
              return new MsgTuple(errorMsg, MsgType.ERROR);            
          }
      }

  }

}

