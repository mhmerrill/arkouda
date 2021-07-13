module FlattenMsg {
  use MultiTypeSymbolTable;
  use MultiTypeSymEntry;
  use Errors;
  use Reflection;
  use Flatten;
  use ServerConfig;
  use SegmentedArray;
  use GenSymIO;
  use Logging;
  use Message;
  
  private config const logLevel = ServerConfig.logLevel;
  const fmLogger = new Logger(logLevel);

  proc segFlattenMsg(cmd: string, payload: string, st: borrowed SymTab): MsgTuple throws {
    var (name, objtype, returnSegsStr, delimJson) = payload.splitMsgToTuple(4);
    const returnSegs: bool = returnSegsStr.toLower() == "true";
    const arr = jsonToPdArray(delimJson, 1);
    const delim: string = arr[arr.domain.low];
    var repMsg: string;
    select objtype {
      when "str" {
        const rSegName = st.nextName();
        const rValName = st.nextName();
        var (stringsName, legacy_placeholder) = name.splitMsgToTuple('+', 2);
        const strings = getSegString(stringsName, st);
        var (off, val, segs) = strings.flatten(delim, returnSegs);
        var stringsObj = getSegString(off, val, st);
        repMsg = "created %s+created bytes.size %t".format(st.attrib(stringsObj.name), stringsObj.nBytes);
        if returnSegs {
          const optName: string = st.nextName();
          st.addEntry(optName, new shared SymEntry(segs));
          repMsg += "+created %s".format(st.attrib(optName));
        }
      } otherwise {
        throw new owned ErrorWithContext("Not implemented for objtype %s".format(objtype),
                                         getLineNumber(),
                                         getRoutineName(),
                                         getModuleName(),
                                         "TypeError");
      }
    }

    fmLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),repMsg);         
    return new MsgTuple(repMsg, MsgType.NORMAL);
  }
}