/* unique finding and counting algorithms
 these are all based on dense histograms and sparse histograms(assoc domains/arrays)

 you could also use a sort if you got into a real bind with really
 large dense ranges of values and large arrays...

 *** need to factor in sparsity estimation somehow ***
 for example if (a.max-a.min > a.size) means that a's are sparse

 */
module UniqueMsg
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

    use Unique;
    
    private config const logLevel = ServerConfig.logLevel;
    const umLogger = new Logger(logLevel);
    
    /* unique take a pdarray and returns a pdarray with the unique values */
    proc uniqueMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (objtype, name, returnCountsStr) = payload.splitMsgToTuple(3);
        // flag to return counts of each unique value
        // same size as unique array
        var returnCounts: bool;
        if returnCountsStr == "True" {returnCounts = true;}
        else if returnCountsStr == "False" {returnCounts = false;}
        else {
            var errorMsg = "Error: %s: %s".format(pn,returnCountsStr);
            umLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);              
            return new MsgTuple(errorMsg, MsgType.ERROR);
        }
        select objtype {
            when "pdarray" {
                // get next symbol name for unique
                var vname = st.nextName();
                // get next symbol anme for counts
                var cname = st.nextName();
                umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                          "cmd: %s name: %s returnCounts: %t: vname: %s cname: %s".format(
                          cmd,name,returnCounts,vname,cname));
        
                var gEnt: borrowed GenSymEntry;
                
                try {  
                    gEnt = st.lookup(name);
                } catch e: Error {
                    throw new owned ErrorWithContext("lookup for %s failed".format(name),
                                       getLineNumber(),
                                       getRoutineName(),
                                       getModuleName(),
                                       "UnknownSymbolError");                
                }

                // the upper limit here is the same as argsort/radixSortLSD_keys
                // check and throw if over memory limit
                overMemLimit(((4 + 1) * gEnt.size * gEnt.itemsize)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
        
                select (gEnt.dtype) {
                    when (DType.Int64) {
                    var e = toSymEntry(gEnt,int);
                
                    /* var eMin:int = min reduce e.a; */
                    /* var eMax:int = max reduce e.a; */
                
                    /* // how many bins in histogram */
                    /* var bins = eMax-eMin+1; */
                    /* umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                           "bins = %t".format(bins)); */

                    /* if (bins <= mBins) { */
                    /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "bins <= %t".format(mBins));*/
                    /*     var (aV,aC) = uniquePerLocHistGlobHist(e.a, eMin, eMax); */
                    /*     st.addEntry(vname, new shared SymEntry(aV)); */
                    /*     if returnCounts {st.addEntry(cname, new shared SymEntry(aC));} */
                    /* } */
                    /* else { */
                    /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                "bins = %t".format(bins));*/
                    /*     var (aV,aC) = uniquePerLocAssocParUnsafeGlobAssocParUnsafe(e.a, eMin, eMax); */
                    /*     st.addEntry(vname, new shared SymEntry(aV)); */
                    /*     if returnCounts {st.addEntry(cname, new shared SymEntry(aC));} */
                    /* } */

                    var (aV,aC) = uniqueSort(e.a);
                    st.addEntry(vname, new shared SymEntry(aV));
                    if returnCounts {
                        st.addEntry(cname, new shared SymEntry(aC));
                    }                  
                }
                otherwise {
                    var errorMsg = notImplementedError("unique",gEnt.dtype);
                    umLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);                
                    return new MsgTuple(errorMsg, MsgType.ERROR);
                }
            }
        
            repMsg = "created " + st.attrib(vname);

            if returnCounts {
                repMsg += " +created " + st.attrib(cname);
            }
            umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);  
            return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          when "str" {
              // TODO remove legacy_placeholder
              var (names1, legacy_placeholder) = name.splitMsgToTuple('+', 2);
              var str = getSegString(names1, st);

              /*
               * The upper limit here is the similar to argsort/radixSortLSD_keys, but with 
               * a few more scratch arrays check and throw if over memory limit.
               */
              overMemLimit((8 * str.size * 8)
                         + (2 * here.maxTaskPar * numLocales * 2**16 * 8));
              var (uo, uv, c, inv) = uniqueGroup(str);
              var myStr = getSegString(uo, uv, st);
              // TODO remove second, legacy call to st.attrib(myStr.name)
              repMsg = "created %s+created bytes.size %t".format(st.attrib(myStr.name), myStr.nBytes);

              if returnCounts {
                  var countName = st.nextName();
                  st.addEntry(countName, new shared SymEntry(c));
                  repMsg += " +created " + st.attrib(countName);
              }

              umLogger.debug(getModuleName(), getRoutineName(), getLineNumber(), repMsg);
              return new MsgTuple(repMsg, MsgType.NORMAL);
          }
          otherwise { 
             var errorMsg = notImplementedError(Reflection.getRoutineName(), objtype);
             umLogger.error(getModuleName(), getRoutineName(), getLineNumber(), errorMsg);
             return new MsgTuple(errorMsg, MsgType.ERROR);              
           }
        }
    }
    
    /* value_counts takes a pdarray and returns two pdarrays unique values and counts for each value */
    proc value_countsMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
        param pn = Reflection.getRoutineName();
        var repMsg: string; // response message
        // split request into fields
        var (name) = payload.splitMsgToTuple(1);

        // get next symbol name
        var vname = st.nextName();
        var cname = st.nextName();
        umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                       "cmd: %s name: %s vname: %s cname: %s".format(cmd, name, vname, cname));

        var gEnt: borrowed GenSymEntry;
        
        try {  
            gEnt = st.lookup(name);
        } catch e: Error {
            throw new owned ErrorWithContext("lookup for %s failed".format(name),
                               getLineNumber(),
                               getRoutineName(),
                               getModuleName(),
                               "UnknownSymbolError");    
        }

        select (gEnt.dtype) {
            when (DType.Int64) {
                var e = toSymEntry(gEnt,int);
                /* var eMin:int = min reduce e.a; */
                /* var eMax:int = max reduce e.a; */

                /* // how many bins in histogram */
                /* var bins = eMax-eMin+1; */
                /* umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins = %t".format(bins));*/

                /* if (bins <= mBins) { */
                /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins <= %t".format(mBins));*/
                /*     var (aV,aC) = uniquePerLocHistGlobHist(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */
                /* else if (bins <= lBins) { */
                /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins <= %t".format(lBins));*/
                /*     var (aV,aC) = uniquePerLocAssocGlobHist(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */
                /* else { */
                /*     umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                         "bins = %t".format(bins));*/
                /*     var (aV,aC) = uniquePerLocAssocGlobAssoc(e.a, eMin, eMax); */
                /*     st.addEntry(vname, new shared SymEntry(aV)); */
                /*     st.addEntry(cname, new shared SymEntry(aC)); */
                /* } */

                var (aV,aC) = uniqueSort(e.a);
                st.addEntry(vname, new shared SymEntry(aV));
                st.addEntry(cname, new shared SymEntry(aC));
            }
            otherwise {
                var errorMsg = notImplementedError(pn,gEnt.dtype);
                umLogger.error(getModuleName(),getRoutineName(),getLineNumber(),errorMsg);
                return new MsgTuple(errorMsg, MsgType.ERROR);                 
            }
        }
        repMsg = "created " + st.attrib(vname) + " +created " + st.attrib(cname);
        umLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);
        return new MsgTuple(repMsg, MsgType.NORMAL);
    }
}