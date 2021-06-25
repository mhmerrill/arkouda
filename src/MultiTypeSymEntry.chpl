
module MultiTypeSymEntry
{
    use ServerConfig;
    use Logging;

    public use NumPyDType;

    public use SymArrayDmap;

    use AryUtil;
    
    /* Casts a GenSymEntry to the specified type and returns it.
       
       :arg gse: generic sym entry
       :type gse: borrowed GenSymEntry

        :arg etype: type for gse to be cast to
        :type etype: type
       */
    inline proc toSymEntry(gse: borrowed GenSymEntry, type etype) {
        return gse.toSymEntry(etype);
    }

    /* This is a dummy class to avoid having to talk about specific
       instantiations of SymEntry. */
    class GenSymEntry
    {
        var dtype: DType; // answer to numpy dtype
        var itemsize: int; // answer to numpy itemsize = num bytes per elt
        var size: int = 0; // answer to numpy size == num elts
        var ndim: int = 1; // answer to numpy ndim == 1-axis for now
        var shape: 1*int = (0,); // answer to numpy shape == 1*int tuple
        
        // not sure yet how to implement numpy data() function

        proc init(type etype, len: int = 0) {
            this.dtype = whichDtype(etype);
            this.itemsize = dtypeSize(this.dtype);
            this.size = len;
            this.shape = (len,);
        }

        /* Cast this `GenSymEntry` to `borrowed SymEntry(etype)`

           This function will halt if the cast fails.

           :arg etype: `SymEntry` type parameter
           :type etype: type
         */
        inline proc toSymEntry(type etype) {
            return try! this :borrowed SymEntry(etype);
        }

        /* 
        Formats and returns data in this entry up to the specified threshold. 
        Arrays of size less than threshold will be printed in their entirety. 
        Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements

            :arg thresh: threshold for data to return
            :type thresh: int

            :arg prefix: String to prepend to the front of the data string
            :type prefix: string

            :arg suffix: String to append to the tail of the data string
            :type suffix: string

            :arg baseFormat: String which represents the base format string for the data type
            :type baseFormat: string

            :returns: s (string) containing the array data
        */
        proc __str__(thresh:int=1, prefix:string="", suffix:string="", baseFormat:string=""): string throws {
            var s = "DType: %s, itemsize: %t, size: %t".format(this.dtype, this.itemsize, this.size);
            return prefix + s + suffix;
        }
    }

    /* Symbol table entry
       Only supports 1-d arrays for now */
    class SymEntry : GenSymEntry
    {
        /*
        generic element type array
        etype is different from dtype (chapel vs numpy)
        */
        type etype;

        /*
        'aD' is the distributed domain for 'a' whose value and type
        are defined by makeDistDom() to support varying distributions
        */
        var aD: makeDistDom(size).type;
        var a: [aD] etype;
        
        /*
        This init takes length and element type

        :arg len: length of array to be allocated
        :type len: int

        :arg etype: type to be instantiated
        :type etype: type
        */
        proc init(len: int, type etype) {
            super.init(etype, len);
            this.etype = etype;
            this.aD = makeDistDom(len);
            // this.a uses default initialization
        }

        /*This init takes an array of a type

        :arg a: array
        :type a: [] ?etype
        */
        proc init(a: [?D] ?etype) {
            super.init(etype, D.size);
            this.etype = etype;
            this.aD = D;
            this.a = a;
        }

        /*
        Verbose flag utility method
        */
        proc postinit() {
            //if v {write("aD = "); printOwnership(this.a);}
        }
        /*
        Verbose flag utility method
        */
        proc deinit() {
            if logLevel == LogLevel.DEBUG {writeln("deinit SymEntry");try! stdout.flush();}
        }
        
        override proc writeThis(f) throws {
          use Reflection;
          proc writeField(f, param i) throws {
            if !isArray(getField(this, i)) {
              f <~> getFieldName(this.type, i) <~> " = " <~> getField(this, i):string;
            } else {
              f <~> getFieldName(this.type, i) <~> " = " <~> formatAry(getField(this, i));
            }
          }

          super.writeThis(f);
          f <~> " {";
          param nFields = numFields(this.type);
          for param i in 0..nFields-2 {
            writeField(f, i);
            f <~> ", ";
          }
          writeField(f, nFields-1);
          f <~> "}";
        }

        /*
        Formats and returns data in this entry up to the specified threshold. 
        Arrays of size less than threshold will be printed in their entirety. 
        Arrays of size greater than or equal to threshold will print the first 3 and last 3 elements

            :arg thresh: threshold for data to return
            :type thresh: int

            :arg prefix: String to pre-pend to the front of the data string
            :type prefix: string

            :arg suffix: String to append to the tail of the data string
            :type suffix: string

            :arg baseFormat: String which represents the base format string for the data type
            :type baseFormat: string

            :returns: s (string) containing the array data
        */
        override proc __str__(thresh:int=6, prefix:string = "[", suffix:string = "]", baseFormat:string = "%t"): string throws {
            var s:string = "";
            if (this.size == 0) {
                s =  ""; // Unnecessary, but left for clarity
            } else if (this.size < thresh || this.size <= 6) {
                for i in 0..(this.size-2) {s += try! baseFormat.format(this.a[i]) + " ";}
                s += try! baseFormat.format(this.a[this.size-1]);
            } else {
                var b = baseFormat + " " + baseFormat + " " + baseFormat + " ... " +
                            baseFormat + " " + baseFormat + " " + baseFormat;
                s = try! b.format(
                            this.a[0], this.a[1], this.a[2],
                            this.a[this.size-3], this.a[this.size-2], this.a[this.size-1]);
            }
            
            if (bool == this.etype) {
                s = s.replace("true","True");
                s = s.replace("false","False");
            }

            return prefix + s + suffix;
        }
    }

}
