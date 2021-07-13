use TestBase;
use UnitTest;

use SegmentedMsg;
use List;

config const N: int = 1_000;
config const MINLEN: int = 6;
config const MAXLEN: int = 30;
config const SUBSTRING: string = "hi";
config const DEBUG = false;
const nb_str:string = b"\x00".decode(); // create null_byte string
const nb_byt:bytes = b"\x00"; // create null_byte


proc make_strings(substr, n, minLen, maxLen, characters, st) {
  const nb = substr.numBytes;
  const sbytes: [0..#nb] uint(8) = for b in substr.chpl_bytes() do b;
  var (segs, vals) = newRandStringsUniformLength(n, minLen, maxLen, characters);
  var strings = getSegString(segs, vals, st);
  var lengths = strings.getLengths() - 1;
  var r: [segs.domain] int;
  fillInt(r, 0, 100);
  var splits = (r % 3);
  /* var present = (r < 5) & (lengths >= nb); */
  /* present[present.domain.high] = true; */
  /* present[present.domain.low] = true; */
  forall (rn, o, l, s) in zip(r, segs, lengths, splits) {
    if l >= nb {
      if (s == 1) || ((s == 2) && (l < 2*(nb+1))) {
        if l == nb {
          vals[{o..#nb}] = sbytes;
        } else {
          const i = rn % (l - nb);
          vals[{(o+i)..#nb}] = sbytes;
        }
      } else if s == 2 {
        const i = rn % ((l/2) - nb);
        vals[{(o+i)..#nb}] = sbytes;
        const j = i + (l/2);
        vals[{(o+j)..#nb}] = sbytes;
      }
    }
  }
  
  var strings2 = getSegString(segs, vals, st): shared SegString;
  return (splits, strings2);
}

proc testPeel(substr:string, n:int, minLen:int, maxLen:int, characters:charSet = charSet.Uppercase) throws {
  var st = new owned SymTab();
  var d: Diags;
  writeln("Generating random strings..."); stdout.flush();
  d.start();
  var (answer, strings) = make_strings(substr, n, minLen, maxLen, characters, st);
  d.stop("make_strings");
  const lengths = strings.getLengths();
  var allSuccess = true;
  for param times in 1..2 {
    for param id in 0..1 {
      param includeDelimiter:bool = id > 0;
      for param kp in 0..1 {
        param keepPartial:bool = kp > 0;
        for param l in 0..1 {
          param left:bool = l > 0;
          writeln("strings.peel(%s, %i, includeDelimiter=%t, keepPartial=%t, left=%t)".format(substr, times, includeDelimiter, keepPartial, left));
          d.start();
          var (leftOffsets, leftVals, rightOffsets, rightVals) = strings.peel(substr, times, includeDelimiter, keepPartial, left);
          d.stop("peel");
          var lstr = getSegString(leftOffsets, leftVals, st);
          var rstr = getSegString(rightOffsets, rightVals, st);
          if DEBUG {
            var llen = lstr.getLengths();
            var rlen = rstr.getLengths();
            var badLen = + reduce ((llen <= 0) | (rlen <= 0));
            writeln("Lengths <= 0: %t".format(badLen));
            if badLen > 0 {
              var n = 0;
              for (i, ll, rl, ol) in zip(lstr.offsets.aD, llen, rlen, lengths) {
                if (ll <= 0) {
                  n += 1;
                  writeln("%i: %s (%i) -> <bad> (%i) | %s (%i)".format(i, strings[i], ol, ll, rstr[i], rl));
                } else if (rl <= 0) {
                  n += 1;
                  writeln("%i: %s (%i) -> %s (%i) | <bad> (%i)".format(i, strings[i], ol, lstr[i], ll, rl));
                }
                if n >= 5 {
                  stdout.flush();
                  break;
                }
              }
            }
          } 
          writeSegString("lstr", lstr);
          const delim = if includeDelimiter then "" else substr;
          var temp: owned SegString?;
          if left {
            var (roundOff, roundVals) = lstr.stick(rstr, delim, true);
            temp = getSegString(roundOff, roundVals, st);
          } else {
            var (roundOff, roundVals) = rstr.stick(lstr, delim, false);
            temp = getSegString(roundOff, roundVals, st);
          }
          var roundTrip: borrowed SegString = temp!;
          var eq = (strings == roundTrip) | (answer < times);
          var success = && reduce eq;
          writeln("Round trip success? >>> %t <<<".format(success));
          allSuccess &&= success;
          if !success {
            var n = 0;
            const rtlen = roundTrip.getLengths();
            for (i, e) in zip(strings.offsets.aD, eq) {
              if !e {
                n += 1;
                writeln("%i: %s (%i) -> %s (%i)".format(i, strings[i], lengths[i], roundTrip[i], rtlen[i]));
              }
              if n >= 5 {
                break;
              }
            }
          }
        }
      }
    }
  }
  writeln("All round trip tests passed? >>> %t <<<".format(allSuccess));
}

proc testMessageLayer(substr, n, minLen, maxLen) throws {
  var st = new owned SymTab();
  var d: Diags;
  writeln("Generating random strings..."); stdout.flush();
  d.start();
  var (answer, strings) = make_strings(substr, n, minLen, maxLen, charSet.Uppercase, st);
  d.stop("make_strings");
  var reqMsg = "peel str %s legacy_placeholder str 1 True True True %jt".format(strings.name, [substr]);
  writeReq(reqMsg);
  var repMsg = segmentedPeelMsg(cmd="segmentedPeel", payload=reqMsg, st).msg;
  writeRep(repMsg);
  var (loAttribs,lvAttribs,roAttribs,rvAttribs) = repMsg.splitMsgToTuple('+', 4);
  var loname = parseName(loAttribs);
  var roname = parseName(roAttribs);
  reqMsg = "stick str %s %s str %s %s False %jt".format(loname, "legacy_placeholder", roname, "legacy_placeholder", [""]);
  writeReq(reqMsg);
  repMsg = segBinopvvMsg(cmd="segBinopvv", payload=reqMsg, st).msg;
  writeRep(repMsg);
  //TODO remove legacy_placeholder
  var (rtoAttribs,legacy_placeholder) = repMsg.splitMsgToTuple('+', 2);
  var rtoname = parseName(rtoAttribs);
  var roundTrip = getSegString(rtoname, st);
  var success = && reduce (strings == roundTrip);
  writeln("Round trip successful? >>> %t <<<".format(success));
}

/**
 * Test case when the delimiter is longer than the last string item
 * See Issue #838
 */
proc testPeelLongDelimiter(test: borrowed Test) throws {

  const d = "----------"; // 10 dashes for delimiter
  var s = nb_str.join("abc%sxyz".format(d), "small%sdog".format(d), "blue%shat".format(d), "last") + nb_str;
  test.assertTrue(59 == s.size);

  var st = new owned SymTab();
  var strings = makeSegArrayFromString(s, st);

  ///////////////////////////// 
  // First test: peel from the left
  var (leftOffsets, leftVals, rightOffsets, rightVals) = strings.peel(d, 1, false, false, true); // from left
  compareArrays(test, [0, 4, 10, 15], leftOffsets);
  compareArrays(test, "abc\x00small\x00blue\x00\x00".encode().bytes(), leftVals);
  compareArrays(test, [0, 4, 8, 12], rightOffsets);
  compareArrays(test, "xyz\x00dog\x00hat\x00last\x00".encode().bytes(), rightVals);

  ///////////////////////////// 
  // Second test: peel from the right
  var (leftOffsetsR, leftValsR, rightOffsetsR, rightValsR) = strings.peel(d, 1, false, false, false); // from right
  test.assertTrue(4 == leftOffsetsR.size && 4 == rightOffsetsR.size);
  compareArrays(test, [0, 4, 10, 15], leftOffsetsR, "First from right test, leftOffsetsR:", false);
  compareArrays(test, "abc\x00small\x00blue\x00last\x00".encode().bytes(), leftValsR, "First fromRight test, leftValsR:", false);
  compareArrays(test, [0, 4, 8, 12], rightOffsetsR, "First fromRight test, rightOffsetsR:", false);
  compareArrays(test, "xyz\x00dog\x00hat\x00\x00".encode().bytes(), rightValsR, "First fromRight test, rightValsR:", false);

  ///////////////////////////// 
  // Run one more with a different size array, from Left
  // Note: reusing parts here causes some overflow issues, so use new vars as appropriate
  s = nb_str.join("abc%sxyz".format(d), "small%sdog".format(d), "last") + nb_str;
  test.assertTrue(41 == s.size);
  strings = makeSegArrayFromString(s, st);
  var (leftOffsets2, leftVals2, rightOffsets2, rightVals2) = strings.peel(d, 1, false, false, true); // from left
  compareArrays(test, [0, 4, 10], leftOffsets2, "fromLeft, leftOffsets2:", false);
  compareArrays(test, "abc\x00small\x00\x00".encode().bytes(), leftVals2, "fromLeft, leftVals2:", false);
  compareArrays(test, [0, 4, 8], rightOffsets2, "fromLeft, rightOffsets2:", false);
  compareArrays(test, "xyz\x00dog\x00last\x00".encode().bytes(), rightVals2, "fromLeft, rightVals2:", false);
  
  ///////////////////////////// 
  // Fourth test: peel from the Right
  var (leftOffsets2R, leftVals2R, rightOffsets2R, rightVals2R) = strings.peel(d, 1, false, false, false); // from right
  compareArrays(test, [0, 4, 10], leftOffsets2R, "fromLeft, leftOffsets2R:", false);
  compareArrays(test, "abc\x00small\x00last\x00".encode().bytes(), leftVals2R, "fromLeft, leftVals2R:", false);
  compareArrays(test, [0, 4, 8], rightOffsets2R, "fromLeft, rightOffsets2R:", false);
  compareArrays(test, "xyz\x00dog\x00\x00".encode().bytes(), rightVals2R, "fromLeft, rightVals2R:", false);
}

proc testPeelIncludeDelimiter(test: borrowed Test) throws {
  const d = "----------"; // 10 dashes for delimiter
  var s = nb_str.join("abc%sxyz".format(d), "small%sdog".format(d), "blue%shat".format(d), "last") + nb_str;
  test.assertTrue(59 == s.size);

  var st = new owned SymTab();
  var strings = makeSegArrayFromString(s, st);

  ///////////////////////////// 
  // First test: peel from the left
  var (leftOffsets, leftVals, rightOffsets, rightVals) = strings.peel(d, 1, true, false, true); // from left
  compareArrays(test, [0, 14, 30, 45], leftOffsets, "fromLeft::leftOffsets:", false);
  compareArrays(test, "abc----------\x00small----------\x00blue----------\x00\x00".encode().bytes(), leftVals, "fromLeft::leftVals:", false);
  compareArrays(test, [0, 4, 8, 12], rightOffsets, "fromLeft::rightOffsets:", false);
  compareArrays(test, "xyz\x00dog\x00hat\x00last\x00".encode().bytes(), rightVals, "fromLeft::rightVals:", false);
   
  ///////////////////////////// 
  // Second test: peel from the right
  var (leftOffsetsR, leftValsR, rightOffsetsR, rightValsR) = strings.peel(d, 1, true, false, false); // from Right
  compareArrays(test, [0, 4, 10, 15], leftOffsetsR, "fromRight::leftOffsetsR:", false);
  compareArrays(test, "abc\x00small\x00blue\x00last\x00".encode().bytes(), leftValsR, "fromRight::leftValsR:", false);
  compareArrays(test, [0, 14, 28, 42], rightOffsetsR, "fromRight::rightOffsetsR:", false);
  compareArrays(test, "----------xyz\x00----------dog\x00----------hat\x00\x00".encode().bytes(), rightValsR, "fromRight::rightValsR:", false);
}

proc compareArrays(test: borrowed Test, expected, actual, msg:string="", debug:bool = false) {
  for (ex, ac) in zip(expected, actual) {
    if(debug) {
      writeln("msg:", msg, " expected:", ex, " - actual:", ac);
    }
    test.assertTrue(ex == ac);
  }
}

/**
 * Internal test utility to make a SegString object from a string.
 * The string should be a concatenation of strings split by null bytes.
 */
proc makeSegArrayFromString(s:string, st) throws {
  // build offsets from null byte positions
  var offset_list = new list(int);
  var bytes_list = new list(uint(8));
  offset_list.append(0); // first string starts at zero
  const length = s.size;
  for (i, b) in zip(0.., s.encode().items()){
    if (nb_byt == b && (i+1) != length) {
      offset_list.append(i+1);
    }
    bytes_list.append(b.toByte());
  }
  if (bytes_list.last() != nb_byt.toByte()) {
    bytes_list.append(nb_byt.toByte());
  }

  return getSegString(offset_list.toArray(), bytes_list.toArray(), st);
}

proc main() {
  var t = new Test();
  try! testPeel(SUBSTRING, N, MINLEN, MAXLEN);
  try! testMessageLayer(SUBSTRING, N, MINLEN, MAXLEN);
  try! testPeelLongDelimiter(t);
  try! testPeelIncludeDelimiter(t);
}
  
