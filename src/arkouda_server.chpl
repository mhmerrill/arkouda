/* arkouda server
backend chapel program to mimic ndarray from numpy
This is the main driver for the arkouda server */

use FileIO;
use Security;
use ServerConfig;
use Time only;
use ZMQ only;
use Memory;
use FileSystem;
use IO;
use Logging;
use Path;
use MultiTypeSymbolTable;
use MultiTypeSymEntry;
use MsgProcessing;
use GenSymIO;
use Reflection;
use SymArrayDmap;
use ServerErrorStrings;
use Message;

private config const logLevel = ServerConfig.logLevel;
const asLogger = new Logger(logLevel);

proc initArkoudaDirectory() {
    var arkDirectory = '%s%s%s'.format(here.cwd(), pathSep,'.arkouda');
    initDirectory(arkDirectory);
    return arkDirectory;
}

proc main() {
 
    proc printServerSplashMessage(token: string, arkDirectory: string) throws {
        var verMessage = "arkouda server version = %s".format(arkoudaVersion);
        var dirMessage = ".arkouda directory %s".format(arkDirectory);
        var numLocMessage = "number of locales = %i".format(numLocales);
        var memLimMessage =  "per locale memory limit = %i".format(getMemLimit());
        var memUsedMessage = "bytes of memory used = %i".format(getMemUsed());
        var serverMessage: string;
    
        const buff = '                         ';
    
        proc adjustMsg(msg: string) throws {
            if msg.size % 2 != 0 {
                return msg + ' ';
            } else {
                return msg;
            }   
        }
    
        proc generateBuffer(longSegment: string, shortSegment: string) : string {
            var buffSize = (longSegment.size - shortSegment.size)/2 - 2;
            var buffer: string;
            var counter = 0;
        
            while counter <= buffSize {
                buffer+=' ';
                counter+=1;
            }           
            return buffer;
        }
    
        if token.isEmpty() {
            serverMessage = "server listening on tcp://%s:%t".format(serverHostname, 
                                                                 ServerPort);
        } else {
            serverMessage = "server listening on tcp://%s:%i?token=%s".format(serverHostname, 
                                                                 ServerPort, token);
        }
        
        serverMessage = adjustMsg(serverMessage);      
        serverMessage = "%s %s %s".format(buff,serverMessage,buff);
        
        var vBuff = generateBuffer(serverMessage,verMessage);
        verMessage = adjustMsg(verMessage);
        verMessage = "*%s %s %s*".format(vBuff,verMessage,vBuff);

        var mlBuff = generateBuffer(serverMessage,memLimMessage);
        memLimMessage = adjustMsg(memLimMessage);
        memLimMessage = "*%s %s %s*".format(mlBuff,memLimMessage,mlBuff);

        var muBuff = generateBuffer(serverMessage,memUsedMessage);
        memUsedMessage = adjustMsg(memUsedMessage);
        memUsedMessage = "*%s %s %s*".format(muBuff,memUsedMessage,muBuff);
        
        var blankBuffer: string;
        var counter = 0;
        
        while counter < serverMessage.size {
            blankBuffer+=' ';
            counter+=1;
        }

        var blankLine = '*%s*'.format(blankBuffer);
        
        var tag = '*';
        counter = 0;
        
        while counter <= serverMessage.size {
            tag+='*';
            counter+=1;
        }

        writeln();
        writeln();
        writeln(tag);
        writeln(tag);
        writeln(blankLine);
        writeln('*%s*'.format(serverMessage));
        writeln(verMessage);
        if (memTrack) {
            writeln(numLocMessage);
            writeln(memLimMessage);
            writeln(memUsedMessage);
        }
        writeln(blankLine);
        writeln(tag);
        writeln(tag);
        writeln();
        writeln();
        stdout.flush();
    }

    const arkDirectory = initArkoudaDirectory();

    var st = new owned SymTab();
    var shutdownServer = false;
    var serverToken : string;
    var serverMessage : string;

    // create and connect ZMQ socket
    var context: ZMQ.Context;
    var socket : ZMQ.Socket = context.socket(ZMQ.REP);

    // configure token authentication if applicable
    if authenticate {
        serverToken = getArkoudaToken('%s%s%s'.format(arkDirectory, pathSep, 'tokens.txt'));
    }

    printServerSplashMessage(serverToken,arkDirectory);

    socket.bind("tcp://*:%t".format(ServerPort));
    
    asLogger.debug(getModuleName(), getRoutineName(), getLineNumber(),
                               "initialized the .arkouda directory %s".format(arkDirectory));
    
    createServerConnectionInfo();

    var reqCount: int = 0;
    var repCount: int = 0;

    var t1 = new Time.Timer();
    t1.clear();
    t1.start();

    /*
    Following processing of incoming message, sends a message back to the client.

    :arg repMsg: either a string or bytes to be sent
    */
    proc sendRepMsg(repMsg: ?t) throws where t==string || t==bytes {
        repCount += 1;
        if trace {
          if t==bytes {
              asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                                        "repMsg: <binary-data>");
          } else {
              asLogger.info(getModuleName(),getRoutineName(),getLineNumber(), 
                                                        "repMsg: %s".format(repMsg));
          }
        }
        socket.send(repMsg);
    }

    /*
    Compares the token submitted by the user with the arkouda_server token. If the
    tokens do not match, or the user did not submit a token, an ErrorWithMsg is thrown.    

    :arg token: the submitted token string
    */
    proc authenticateUser(token : string) throws {
        if token == 'None' || token.isEmpty() {
            throw new owned ErrorWithMsg("Error: access to arkouda requires a token");
        }
        else if serverToken != token {
            throw new owned ErrorWithMsg("Error: token %s does not match server token, check with server owner".format(token));
        }
    } 

    /*
     * Converts the incoming request JSON string into RequestMsg object.
     */
    proc extractRequest(request : string) : RequestMsg throws {
        var rm = new RequestMsg();
        deserialize(rm, request);
        return rm;
    }
    
    /*
    Sets the shutdownServer boolean to true and sends the shutdown command to socket,
    which stops the arkouda_server listener thread and closes socket.
    */
    proc shutdown(user: string) {
        shutdownServer = true;
        repCount += 1;
        socket.send(serialize(msg="shutdown server (%i req)".format(repCount), 
                         msgType=MsgType.NORMAL,msgFormat=MsgFormat.STRING, user=user));
    }
    
    while !shutdownServer {
        // receive message on the zmq socket
        var reqMsgRaw = socket.recv(bytes);

        reqCount += 1;

        var s0 = t1.elapsed();
        
        /*
         * Separate the first tuple, which is a string binary containing the JSON binary
         * string encapsulating user, token, cmd, message format and args from the 
         * remaining payload.
         */
        var (rawRequest, payload) = reqMsgRaw.splitMsgToTuple(b"BINARY_PAYLOAD",2);
        var user, token, cmd: string;

        // parse requests, execute requests, format responses
        try {
            /*
             * Decode the string binary containing the JSON-formatted request string. 
             * If there is an error, discontinue processing message and send an error
             * message back to the client.
             */
            var request : string;

            try! {
                request = rawRequest.decode();
            } catch e: DecodeError {
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                       "illegal byte sequence in command: %t".format(
                                          rawRequest.decode(decodePolicy.replace)));
                sendRepMsg(serialize(msg=unknownError(e.message()),msgType=MsgType.ERROR,
                                                 msgFormat=MsgFormat.STRING, user="Unknown"));
            }

            // deserialize the decoded, JSON-formatted cmdStr into a RequestMsg
            var msg: RequestMsg  = extractRequest(request);
            user   = msg.user;
            token  = msg.token;
            cmd    = msg.cmd;
            var format = msg.format;
            var args   = msg.args;

            /*
             * If authentication is enabled with the --authenticate flag, authenticate
             * the user which for now consists of matching the submitted token
             * with the token generated by the arkouda server
             */ 
            if authenticate {
                authenticateUser(token);
            }

            if (trace) {
              try {
                if (cmd != "array") {
                  asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                                     ">>> %t %t".format(cmd, args));
                } else {
                  asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
                                                     ">>> %s [binary data]".format(cmd));
                }
              } catch {
                // No action on error
              }
            }

            // If cmd is shutdown, don't bother generating a repMsg
            if cmd == "shutdown" {
                shutdown(user=user);
                if (trace) {
                    asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                                         "<<< shutdown initiated by %s took %.17r sec".format(user, 
                                                   t1.elapsed() - s0));
                }
                break;
            }

            /*
             * Declare the repTuple and binaryRepMsg variables, one of which is processed, if 
             * applicable, and sent to sendRepMsg depending upon whether a string (repTuple)
             * or bytes (binaryRepMsg) is to be returned.
             */
            var binaryRepMsg: bytes;
            var repTuple: MsgTuple;

            select cmd
            {
                when "array"             {repTuple = arrayMsg(cmd, payload, st);}
                when "tondarray"         {binaryRepMsg = tondarrayMsg(cmd, args, st);}
                when "cast"              {repTuple = castMsg(cmd, args, st);}
                when "mink"              {repTuple = minkMsg(cmd, args, st);}
                when "maxk"              {repTuple = maxkMsg(cmd, args, st);}
                when "intersect1d"       {repTuple = intersect1dMsg(cmd, args, st);}
                when "setdiff1d"         {repTuple = setdiff1dMsg(cmd, args, st);}
                when "setxor1d"          {repTuple = setxor1dMsg(cmd, args, st);}
                when "union1d"           {repTuple = union1dMsg(cmd, args, st);}
                when "segStr-assemble"   {repTuple = assembleStringsMsg(cmd, args, st);}
                when "segStr-tondarray"  {binaryRepMsg = segStrTondarrayMsg(cmd, args, st);}
                when "segmentLengths"    {repTuple = segmentLengthsMsg(cmd, args, st);}
                when "segmentedHash"     {repTuple = segmentedHashMsg(cmd, args, st);}
                when "segmentedEfunc"    {repTuple = segmentedEfuncMsg(cmd, args, st);}
                when "segmentedPeel"     {repTuple = segmentedPeelMsg(cmd, args, st);}
                when "segmentedIndex"    {repTuple = segmentedIndexMsg(cmd, args, st);}
                when "segmentedBinopvv"  {repTuple = segBinopvvMsg(cmd, args, st);}
                when "segmentedBinopvs"  {repTuple = segBinopvsMsg(cmd, args, st);}
                when "segmentedGroup"    {repTuple = segGroupMsg(cmd, args, st);}
                when "segmentedIn1d"     {repTuple = segIn1dMsg(cmd, args, st);}
                when "segmentedFlatten"  {repTuple = segFlattenMsg(cmd, args, st);}
                when "lshdf"             {repTuple = lshdfMsg(cmd, args, st);}
                
                // DEPRECATED all client paths point to readAllHdfMsg
                when "readhdf"           {repTuple = readhdfMsg(cmd, args, st);}
                
                when "readAllHdf"        {repTuple = readAllHdfMsg(cmd, args, st);}
                when "tohdf"             {repTuple = tohdfMsg(cmd, args, st);}
                when "create"            {repTuple = createMsg(cmd, args, st);}
                when "delete"            {repTuple = deleteMsg(cmd, args, st);}
                when "binopvv"           {repTuple = binopvvMsg(cmd, args, st);}
                when "binopvs"           {repTuple = binopvsMsg(cmd, args, st);}
                when "binopsv"           {repTuple = binopsvMsg(cmd, args, st);}
                when "opeqvv"            {repTuple = opeqvvMsg(cmd, args, st);}
                when "opeqvs"            {repTuple = opeqvsMsg(cmd, args, st);}
                when "efunc"             {repTuple = efuncMsg(cmd, args, st);}
                when "efunc3vv"          {repTuple = efunc3vvMsg(cmd, args, st);}
                when "efunc3vs"          {repTuple = efunc3vsMsg(cmd, args, st);}
                when "efunc3sv"          {repTuple = efunc3svMsg(cmd, args, st);}
                when "efunc3ss"          {repTuple = efunc3ssMsg(cmd, args, st);}
                when "reduction"         {repTuple = reductionMsg(cmd, args, st);}
                when "countReduction"    {repTuple = countReductionMsg(cmd, args, st);}
                when "findSegments"      {repTuple = findSegmentsMsg(cmd, args, st);}
                when "segmentedReduction"{repTuple = segmentedReductionMsg(cmd, args, st);}
                when "broadcast"         {repTuple = broadcastMsg(cmd, args, st);}
                when "arange"            {repTuple = arangeMsg(cmd, args, st);}
                when "linspace"          {repTuple = linspaceMsg(cmd, args, st);}
                when "randint"           {repTuple = randintMsg(cmd, args, st);}
                when "randomNormal"      {repTuple = randomNormalMsg(cmd, args, st);}
                when "randomStrings"     {repTuple = randomStringsMsg(cmd, args, st);}
                when "histogram"         {repTuple = histogramMsg(cmd, args, st);}
                when "in1d"              {repTuple = in1dMsg(cmd, args, st);}
                when "unique"            {repTuple = uniqueMsg(cmd, args, st);}
                when "value_counts"      {repTuple = value_countsMsg(cmd, args, st);}
                when "set"               {repTuple = setMsg(cmd, args, st);}
                when "info"              {repTuple = infoMsg(cmd, args, st);}
                when "str"               {repTuple = strMsg(cmd, args, st);}
                when "repr"              {repTuple = reprMsg(cmd, args, st);}
                when "[int]"             {repTuple = intIndexMsg(cmd, args, st);}
                when "[slice]"           {repTuple = sliceIndexMsg(cmd, args, st);}
                when "[pdarray]"         {repTuple = pdarrayIndexMsg(cmd, args, st);}
                when "[int]=val"         {repTuple = setIntIndexToValueMsg(cmd, args, st);}
                when "[pdarray]=val"     {repTuple = setPdarrayIndexToValueMsg(cmd, args, st);}
                when "[pdarray]=pdarray" {repTuple = setPdarrayIndexToPdarrayMsg(cmd, args, st);}
                when "[slice]=val"       {repTuple = setSliceIndexToValueMsg(cmd, args, st);}
                when "[slice]=pdarray"   {repTuple = setSliceIndexToPdarrayMsg(cmd, args, st);}
                when "argsort"           {repTuple = argsortMsg(cmd, args, st);}
                when "coargsort"         {repTuple = coargsortMsg(cmd, args, st);}
                when "concatenate"       {repTuple = concatenateMsg(cmd, args, st);}
                when "sort"              {repTuple = sortMsg(cmd, args, st);}
                when "joinEqWithDT"      {repTuple = joinEqWithDTMsg(cmd, args, st);}
                when "getconfig"         {repTuple = getconfigMsg(cmd, args, st);}
                when "getmemused"        {repTuple = getmemusedMsg(cmd, args, st);}
                when "register"          {repTuple = registerMsg(cmd, args, st);}
                when "attach"            {repTuple = attachMsg(cmd, args, st);}
                when "unregister"        {repTuple = unregisterMsg(cmd, args, st);}
                when "clear"             {repTuple = clearMsg(cmd, args, st);}               
                when "connect" {
                    if authenticate {
                        repTuple = new MsgTuple("connected to arkouda server tcp://*:%i as user %s with token %s".format(
                                                          ServerPort,user,token), MsgType.NORMAL);
                    } else {
                        repTuple = new MsgTuple("connected to arkouda server tcp://*:%i".format(ServerPort), 
                                                                                        MsgType.NORMAL);
                    }
                }
                when "disconnect" {
                    repTuple = new MsgTuple("disconnected from arkouda server tcp://*:%i".format(ServerPort), 
                                                                   MsgType.NORMAL);
                }
                when "noop" {
                    repTuple = new MsgTuple("noop", MsgType.NORMAL);
                }
                when "ruok" {
                    repTuple = new MsgTuple("imok", MsgType.NORMAL);
                }
                otherwise {
                    repTuple = new MsgTuple("Unrecognized command: %s".format(cmd), MsgType.ERROR);
                    asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),repTuple.msg);
                }
            }

            /*
             * 1. Determine if the reply message is binary or a string via the repTuple.msg attribute
             * 2. If a string, invoke serialize to generate a JSON-formatted reply string
             * 3. Invoke the sendRepMsg function
             */          
            if repTuple.msg.isEmpty() {
                // Since the repTuple.msg attribute is empty, this is a binary reply message
                sendRepMsg(binaryRepMsg);
            } else {
                sendRepMsg(serialize(msg=repTuple.msg,msgType=repTuple.msgType,
                                                              msgFormat=MsgFormat.STRING, user=user));
            }

            /*
             * log that the request message has been handled and reply message has been sent along with 
             * the time to do so
             */
            if trace {
                asLogger.info(getModuleName(),getRoutineName(),getLineNumber(), 
                                              "<<< %s took %.17r sec".format(cmd, t1.elapsed() - s0));
            }
            if (trace && memTrack) {
                asLogger.info(getModuleName(),getRoutineName(),getLineNumber(),
                    "bytes of memory used after command %t".format(getMemUsed():uint * numLocales:uint));
            }
        } catch (e: ErrorWithMsg) {
            // Generate a ReplyMsg of type ERROR and serialize to a JSON-formatted string
            sendRepMsg(serialize(msg=e.msg,msgType=MsgType.ERROR, msgFormat=MsgFormat.STRING, 
                                                        user=user));
            if trace {
                asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                    "<<< %s resulted in error %s in  %.17r sec".format(cmd, e.msg, t1.elapsed() - s0));
            }
        } catch (e: Error) {
            // Generate a ReplyMsg of type ERROR and serialize to a JSON-formatted string
            var errorMsg = e.message();
            
            if errorMsg.isEmpty() {
                errorMsg = "unexpected error";
            }

            sendRepMsg(serialize(msg=errorMsg,msgType=MsgType.ERROR, 
                                                         msgFormat=MsgFormat.STRING, user=user));
            if trace {
                asLogger.error(getModuleName(), getRoutineName(), getLineNumber(), 
                    "<<< %s resulted in error: %s in %.17r sec".format(cmd, e.message(),
                                                                                 t1.elapsed() - s0));
            }
        }
    }

    t1.stop();

    deleteServerConnectionInfo();

    asLogger.info(getModuleName(), getRoutineName(), getLineNumber(),
               "requests = %i responseCount = %i elapsed sec = %i".format(reqCount,repCount,
                                                                                 t1.elapsed()));
}

/*
Creates the serverConnectionInfo file on arkouda_server startup
*/
proc createServerConnectionInfo() {
    use IO;
    if !serverConnectionInfo.isEmpty() {
        try! {
            var w = open(serverConnectionInfo, iomode.cw).writer();
            w.writef("%s %t\n", serverHostname, ServerPort);
        }
    }
}

/*
Deletes the serverConnetionFile on arkouda_server shutdown
*/
proc deleteServerConnectionInfo() {
    use FileSystem;
    try {
        if !serverConnectionInfo.isEmpty() {
            remove(serverConnectionInfo);
        }
    } catch fnfe : FileNotFoundError {
        asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "The serverConnectionInfo file was not found %s".format(fnfe.message()));
    } catch e : Error {
        asLogger.error(getModuleName(),getRoutineName(),getLineNumber(),
                              "Error in deleting serverConnectionInfo file %s".format(e.message()));    
    }
}
