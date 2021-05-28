use TestBase;

config const N: int = 10_000;
config const MINLEN: int = 1;
config const MAXLEN: int = 20;
config const SUBSTRING: string = "hi";
config const DEBUG = false;

proc make_strings(substr, n, minLen, maxLen, characters, mode, st) {
  const nb = substr.numBytes;
  const sbytes: [0..#nb] uint(8) = for b in substr.chpl_bytes() do b;
  var (segs, vals) = newRandStringsUniformLength(n, minLen, maxLen, characters);
  var strings = getSegString(segs, vals, st);
  var lengths = strings.getLengths() - 1;
  var r: [segs.domain] int;
  fillInt(r, 0, 100);
  var present = (r < 5) & (lengths >= nb);
  present[present.domain.high] = true;
  present[present.domain.low] = true;
  forall (p, rn, o, l) in zip(present, r, segs, lengths) {
    if p {
      var i: int;
      if mode == SearchMode.contains {
        if l == nb {
          i = 0;
        } else {
          i = rn % (l - nb);
        }
      } else if mode == SearchMode.startsWith {
        i = 0;
      } else if mode == SearchMode.endsWith {
        i = l - nb;
      }
      vals[{(o+i)..#nb}] = sbytes;
    }
  }
  
  var strings2 = getSegString(segs, vals, st): shared SegString;
  return (present, strings2);
}

proc test_search(substr:string, n:int, minLen:int, maxLen:int, characters:charSet = charSet.Uppercase, mode: SearchMode = SearchMode.contains) throws {
  var st = new owned SymTab();
  var d: Diags;
  writeln("Generating random strings..."); stdout.flush();
  d.start();
  var (answer, strings) = make_strings(substr, n, minLen, maxLen, characters, mode, st);
  d.stop("make_strings");
  writeln("Searching for substring..."); stdout.flush();
  d.start();
  var truth = strings.substringSearch(substr, mode);
  d.stop("substringSearch");
  var nFound = + reduce truth;
  if DEBUG && (nFound > 0) {
    writeln("Found %t strings containing %s".format(nFound, substr)); stdout.flush();
    var (mSegs, mVals) = strings[truth];
    var matches = getSegString(mSegs, mVals, st);
    matches.show(5);
    writeln("Seeded with ",  + reduce answer, " values");
  }
  var isMissing = (answer & !truth);
  var missing = + reduce isMissing;
  var isExtra = (truth & !answer);
  var extra = + reduce isExtra;
  if (missing == 0) && (extra == 0) {
    writeln("Perfect match");
  } else {
    writeln("%t missing answers; %t extras".format(missing, extra));
    if DEBUG {
      writeln("missing:");
      var minds = + scan isMissing;
      for i in 1..5 {
        var (blah, idx) = maxloc reduce zip((minds == i), minds.domain);
        if !blah {
          break;
        }
        writeln();
        if idx > minds.domain.low {
          writeln(strings[idx-1]);
        }
        writeln(strings[idx]);
        if idx < minds.domain.high {
          writeln(strings[idx+1]);
        }
      }
      writeln("extras:");
      var einds = + scan isExtra;
      for i in 1..5 {
        var (blah, idx) = maxloc reduce zip((einds == i), einds.domain);
        if !blah {
          break;
        }
        writeln();
        if idx > einds.domain.low {
          writeln(strings[idx-1]);
        }
        writeln(strings[idx]);
        if idx < einds.domain.high {
          writeln(strings[idx+1]);
        }
      }
    }
  }
}

proc main() {
  for mode in (SearchMode.contains, SearchMode.startsWith, SearchMode.endsWith) {
    writeln("\n", mode);
    try! test_search(SUBSTRING, N, MINLEN, MAXLEN, mode=mode);
  }
}
