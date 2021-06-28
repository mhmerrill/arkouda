module SipHash {
  private use CommPrimitives;
  private use AryUtil;
  private use CPtr;
  use ServerConfig;
  use Errors;
  use Reflection;
  use Logging;
  
  param cROUNDS = 2;
  param dROUNDS = 4;

  private config param DEBUG = false;

  const defaultSipHashKey: [0..#16] uint(8) = for i in 0..#16 do i: uint(8);

  private config const logLevel = ServerConfig.logLevel;
  const shLogger = new Logger(logLevel);

  inline proc ROTL(x, b) {
    return (((x) << (b)) | ((x) >> (64 - (b))));
  }

  private inline proc U32TO8_LE(p: [?D] uint(8), v: uint(32)) {
    p[D.low] = v: uint(8);
    p[D.low+1] = (v >> 8): uint(8);
    p[D.low+2] = (v >> 16): uint(8);
    p[D.low+3] = (v >> 24): uint(8);
  }
  
  private inline proc U64TO8_LE(p: [?D] uint(8), v: uint(64)) {
    U32TO8_LE(p[D.low..#4], v: uint(32));
    U32TO8_LE(p[D.low+4..#4], (v >> 32): uint(32));
  }

  private inline proc U8TO64_LE(p: [] uint(8), D): uint(64) {
    return ((p[D.low]: uint(64)) |
            (p[D.low+1]: uint(64) << 8) |
            (p[D.low+2]: uint(64) << 16) |
            (p[D.low+3]: uint(64) << 24) |
            (p[D.low+4]: uint(64) << 32) |
            (p[D.low+5]: uint(64) << 40) |
            (p[D.low+6]: uint(64) << 48) |
            (p[D.low+7]: uint(64) << 56));
  }

  private inline proc U8TO64_LE(p: c_ptr(uint(8))): uint(64) {
    return ((p[0]: uint(64)) |
            (p[1]: uint(64) << 8) |
            (p[2]: uint(64) << 16) |
            (p[3]: uint(64) << 24) |
            (p[4]: uint(64) << 32) |
            (p[5]: uint(64) << 40) |
            (p[6]: uint(64) << 48) |
            (p[7]: uint(64) << 56));
  }

  private inline proc U8TO64_LE(p: [] int, D): uint(64) {
    return ((p[D.low]: uint(64)) |
            (p[D.low+1]: uint(64) << 8) |
            (p[D.low+2]: uint(64) << 16) |
            (p[D.low+3]: uint(64) << 24) |
            (p[D.low+4]: uint(64) << 32) |
            (p[D.low+5]: uint(64) << 40) |
            (p[D.low+6]: uint(64) << 48) |
            (p[D.low+7]: uint(64) << 56));
  }

  private inline proc U8TO64_LE(p: c_ptr(int)): uint(64) {
    return ((p[0]: uint(64)) |
            (p[1]: uint(64) << 8) |
            (p[2]: uint(64) << 16) |
            (p[3]: uint(64) << 24) |
            (p[4]: uint(64) << 32) |
            (p[5]: uint(64) << 40) |
            (p[6]: uint(64) << 48) |
            (p[7]: uint(64) << 56));
  }

  private inline proc byte_reverse(b: uint(64)): uint(64) {
    var c: uint(64);
    c |= (b & 0xff) << 56;
    c |= ((b >> 8) & 0xff) << 48;
    c |= ((b >> 16) & 0xff) << 40;
    c |= ((b >> 24) & 0xff) << 32;
    c |= ((b >> 32) & 0xff) << 24;
    c |= ((b >> 40) & 0xff) << 16;
    c |= ((b >> 48) & 0xff) << 8;
    c |= ((b >> 56) & 0xff);
    return c;
  }
  
  proc sipHash64(msg: [] uint(8), D): uint(64) {
    var (res,_) = computeSipHashLocalized(msg, D, 8);
    return res;
  }

  proc sipHash128(msg: [] uint(8), D): 2*uint(64) {
    return computeSipHashLocalized(msg, D, 16);
  }
  proc sipHash128(msg: [] int, D): 2*uint(64) {
    return computeSipHashLocalized(msg, D, 16);
  }

  private proc computeSipHashLocalized(msg: [] uint(8), D, param outlen: int) {
    if contiguousIndices(msg) {
      ref start = msg[D.low];
      if D.high < D.low {
        return computeSipHash(c_ptrTo(start), 0..#0, outlen);
      }
      ref end = msg[D.high];
      const startLocale = start.locale.id;
      const endLocale = end.locale.id;
      const hereLocale = here.id;
      const l = D.size;
      if startLocale == endLocale {
        if startLocale == hereLocale {
          return computeSipHash(c_ptrTo(start), 0..#l, outlen);
        } else {
          var a = c_malloc(msg.eltType, l);
          GET(a, startLocale, getAddr(start), l);
          var h = computeSipHash(a, 0..#l, outlen);
          c_free(a);
          return h;
        }
      }
    }
    return computeSipHash(msg, D, outlen);
  }
  
  private proc computeSipHashLocalized(msg: [] int, D, param outlen: int) {
    if contiguousIndices(msg) {
      ref start = msg[D.low];
      if D.high < D.low {
        return computeSipHash(c_ptrTo(start), 0..#0, outlen);
      }
      ref end = msg[D.high];
      const startLocale = start.locale.id;
      const endLocale = end.locale.id;
      const hereLocale = here.id;
      const l = D.size;
      if startLocale == endLocale {
        if startLocale == hereLocale {
          return computeSipHash(c_ptrTo(start), 0..#l, outlen);
        } else {
          var a = c_malloc(msg.eltType, l);
          GET(a, startLocale, getAddr(start), l);
          var h = computeSipHash(a, 0..#l, outlen);
          c_free(a);
          return h;
        }
      }
    }
    return computeSipHash(msg, D, outlen);
  }
  private proc computeSipHash(msg, D, param outlen: int) {
    if !((outlen == 8) || (outlen == 16)) {
      compilerError("outlen must be 8 or 16");
    }
    var v0 = 0x736f6d6570736575: uint(64);
    var v1 = 0x646f72616e646f6d: uint(64);
    var v2 = 0x6c7967656e657261: uint(64);
    var v3 = 0x7465646279746573: uint(64);
    const k0 = 0x0706050403020100: uint(64);
    const k1 = 0x0f0e0d0c0b0a0908: uint(64);
    var m: uint(64);
    var i: int;
    const lastPos = D.low + D.size - (D.size % 8);
    // const uint8_t *end = in + inlen - (inlen % sizeof(uint64_t));
    const left: int = D.size & 7;
    // const int left = inlen & 7;
    var b: uint(64) = (D.size: uint(64)) << 56;
    v3 ^= k1;
    v2 ^= k0;
    v1 ^= k1;
    v0 ^= k0;
    
    if (outlen == 16) {
      v1 ^= 0xee;
    }
    
    inline proc SIPROUND() {
        v0 += v1;
        v1 = ROTL(v1, 13);
        v1 ^= v0;
        v0 = ROTL(v0, 32);
        v2 += v3;
        v3 = ROTL(v3, 16);
        v3 ^= v2;
        v0 += v3;
        v3 = ROTL(v3, 21);
        v3 ^= v0;
        v2 += v1;
        v1 = ROTL(v1, 17);
        v1 ^= v2;
        v2 = ROTL(v2, 32);
    }

    inline proc TRACE() {
        if DEBUG {
            try! {
              shLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "%i v0 %016xu".format(D.size, v0));
              shLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "%i v1 %016xu".format(D.size, v1));
              shLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "%i v2 %016xu".format(D.size, v2));
              shLogger.debug(getModuleName(),getRoutineName(),getLineNumber(),
                                                   "%i v3 %016xu".format(D.size, v3));
            }
        }
    }

    for pos in D.low..lastPos-1 by 8 {
        if isSubtype(msg.type, c_ptr) {
          m = U8TO64_LE(msg + pos);
        } else {
          m = U8TO64_LE(msg, pos..#8);
        }
        v3 ^= m;
        TRACE();
        for i in 0..#cROUNDS {
          SIPROUND();
        }

        v0 ^= m;
    }

    if (left == 7) {
        b |= (msg[lastPos+6]: uint(64)) << 48;
}
    if (left >= 6) {
        b |= (msg[lastPos+5]: uint(64)) << 40;
    }
    if (left >= 5) {
        b |= (msg[lastPos+4]: uint(64)) << 32;
}
    if (left >= 4) {
        b |= (msg[lastPos+3]: uint(64)) << 24;
}
    if (left >= 3) {
        b |= (msg[lastPos+2]: uint(64)) << 16;
}
    if (left >= 2) {
        b |= (msg[lastPos+1]: uint(64)) << 8;
}
    if (left >= 1) {
        b |= (msg[lastPos]: uint(64));
        }

    v3 ^= b;

    TRACE();
    for i in 0..#cROUNDS {
      SIPROUND();
    }

    v0 ^= b;

    if (outlen == 16) {
      v2 ^= 0xee;
    } else {
      v2 ^= 0xff;
    }

    TRACE();
    for i in 0..#dROUNDS {
      SIPROUND();
    }

    b = v0 ^ v1 ^ v2 ^ v3;
    const res0 = byte_reverse(b);

    if (outlen == 8) {
        return (res0, 0:uint(64));
    }

    v1 ^= 0xdd;

    TRACE();
    for i in 0..#dROUNDS {
      SIPROUND();
    }
    
    b = v0 ^ v1 ^ v2 ^ v3;

    return  (res0, byte_reverse(b));
  }
}
