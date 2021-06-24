module SACA{
// In this module, different algorithms to construct suffix array are provided
//Nov.15, 2020
//Algorithm 1
// The first algorithm divsufsort is the fastest C codes on suffix array
require "../thirdparty/SA/libdivsufsort/include/config.h";
require "../thirdparty/SA/libdivsufsort/include/divsufsort.h";
require "../thirdparty/SA/libdivsufsort/include/divsufsort_private.h";
require "../thirdparty/SA/libdivsufsort/include/lfs.h";

require "../thirdparty/SA/libdivsufsort/lib/divsufsort.c";
require "../thirdparty/SA/libdivsufsort/lib/sssort.c";
require "../thirdparty/SA/libdivsufsort/lib/trsort.c";
require "../thirdparty/SA/libdivsufsort/lib/utils.c";
extern proc divsufsort(inputstr:[] uint(8),suffixarray:[] int(32),totallen:int(32));

//Another possible SACA algorithm to utilize. 
//require "../thirdparty/SA/SACA-K/saca-k.c";

//extern proc SACA_K(inputstr:[] uint(8), suffixarray:[] uint, n:uint, K:uint,m:uint, level:int);
//void SACA_K(unsigned char *s, unsigned int *SA,
//  unsigned int n, unsigned int K,
//  unsigned int m, int level) ;

//Algorithm 2

// The Chapel version of suffix array construction algorithm using skew algorithm
// Rewrite the algorithm and codes in paper
// "Simple Linear Work Suffix Array Construction" by Juha Karkkainen and Peter Sanders (2003)
// Dec.7, 2020

inline proc leq(a1 :int, a2:int, b1:int, b2:int) // lexicographic order
{  return(a1 < b1 || a1 == b1 && a2 <= b2); 
} // for pairs

inline proc leq(a1 :int, a2:int,a3:int, b1:int, b2:int, b3:int) // lexicographic order
{  return(a1 < b1 || a1 == b1 && leq(a2,a3, b2,b3)); 
} // for pairs

//stably sort a[0..n-1] to b[0..n-1] with keys in 0..K from r
proc radixPass(a:[] int, b:[] int, r:[] uint(8), n:int, K:int )
{  // count occurrences
        var c:[0..K] uint(8); // counter array
        var x:uint(8); 
        var i=0:int;
        var sum=0:int;
        forall x in c do x=0;
        for i in 0..n-1 do  c[r[a[i]]]=c[r[a[i]]]+1;
        var t:uint(8);
        for i in 0..K do  {
             t=c[i];
             c[i]=sum;
             sum+=t;
        }
        for i in 0..n-1  do { 
           b[c[r[a[i]]]] = a[i];
           c[r[a[i]]]=c[r[a[i]]]+1;
        }
} 


//stably sort a[0..n-1] to b[0..n-1] with keys in 0..K from r
//element a[i] is mapping to r[a[i]] and r is the alphabets with K+1 characters.
// a and b are bounded by n in calculation
proc radixPass(a:[] int, b:[] int, r:[] int, n:int, K:int )
{// count occurrences
        var c:[0..K] int; // counter array
        var x:int; 
        var i:int;
        var t:int;
        var sum=0:int;
        forall x in c do x=0;
        // calculate the number of different characters in a
        for i in 0..n-1 do  c[r[a[i]]]=c[r[a[i]]]+1;
        // calculate the presum of c, so c[i] will be the starting position of different characters
        for i in 0..K do  {
             t=c[i];
             c[i]=sum;
             sum+=t;
        }
        // let b[j] store the position of each a[i] based on their order. 
        //The same character but following the previous suffix will be put at the next position.
        for i in 0..n-1 do {
            b[c[r[a[i]]]] = a[i];
            c[r[a[i]]]=c[r[a[i]]]+1;
        }

} 
//stably sort a[0..n-1] to b[0..n-1] with keys in 0..K from r

// find the suffix array SA of s[0..n-1] in {1..K}^n
// require s[n]=s[n+1]=s[n+2]=0, n>=2. So the size of s should be n+3
proc SuffixArraySkew(s:[] int, SA: [] int, n:int, K: int) {
   var  n0=(n+2)/3:int;
   var  n1=(n+1)/3:int;
   var  n2=n/3:int;
   var  n02=n0+n2:int;
   var  n12=n1+n2:int;
//number of elements meet i %3 =0,1, and 2. 
//s[i] is the ith suffix, i in 0..n-1
   var  s12: [0..n02+2] int; 
   s12[n02]= 0;
   s12[n02+1]= 0;
   s12[n02+2]=0;
// Here n02 instead of  n12=n1+n2 is used for the later s0 building based on n1 elements
   var SA12:[0..n02 + 2] int; 
   SA12[n02]=0;
   SA12[n02+1]=0;
   SA12[n02+2]=0;

   var s0:[0.. n0+2] int;
   var SA0:[0..n0+2] int;
   var i=0:int;
   var j=0:int;
   var k=0:int;

// generate positions of mod 1 and mod 2 suffixes
// n0-n1 is used for building s0, s1 has the same number of elements as s0
   for  i in 0.. n+(n0-n1)-1 do { 
       if (i%3 != 0) { 
           s12[j] = i;
           j=j+1;
       }
   }
// lsb radix sort the mod 1 and mod 2 triples
   var tmps:[0..n+2] int;
   forall i in 0..n-2 do tmps[i]=s[i+2];
   radixPass(s12 , SA12, tmps, n02, K);
   forall i in 0..n-1 do tmps[i]=s[i+1];
   radixPass(SA12, s12 , tmps, n02, K);
   radixPass(s12 , SA12, s , n02, K);

// find lexicographic names of triples

   var name = 0:int, c0 = -1:int, c1 = -1:int, c2 = -1:int;

   for i in 0..n02-1 do {
      if (s[SA12[i]] != c0 || s[SA12[i]+1] != c1 || s[SA12[i]+2] != c2)
      {  name=name+1;
         c0 = s[SA12[i]]; 
         c1 = s[SA12[i]+1]; 
         c2 = s[SA12[i]+2]; 
      } 
      if (SA12[i] % 3 == 1) {   
         s12[SA12[i]/3] = name; 
                // mapping the suffix to small alphabets
      } // left half
      else {
         s12[SA12[i]/3 + n0] = name; 
      } // right half
   }  

// recurse if names are not unique
   if (name < n02) {
     SuffixArraySkew(s12, SA12, n02, name);
// store unique names in s12 using the suffix array
     for i in 0..n02-1 do  s12[SA12[i]] = i + 1;
                //restore the value of s12 since we will change its values during the procedure
   }  else // generate the suffix array of s12 directly
   {    for i in 0..n02-1 do  SA12[s12[i] - 1] = i;
          // here SA12 is in fact the ISA array.
   }
// stably sort the mod 0 suffixes from SA12 by their first character
   j=0;
   for i in 0..n02-1 do {
// here in fact we take advantage of the sorted SA12 to just sort s0 once to get its sorted array
// at first we think the postion i%3=1 is the position
       if (SA12[i] < n0) { 
              s0[j] = 3*SA12[i]; 
              j=j+1;
           }
   }
   radixPass(s0, SA0, s, n0, K);

// merge sorted SA0 suffixes and sorted SA12 suffixes
   var p=0:int;// first s0 position
   var t=n0-n1:int;//first s1 position
   k=0;
   var i1:int , j1:int;
   var tmpk:int;
   for tmpk in 0..n-1 do  {
//#define GetI() (SA12[t] < n0 ? SA12[t] * 3 + 1 : (SA12[t] - n0) * 3 + 2)
   proc GetI():int {
      if (SA12[t] < n0 ) { return SA12[t] * 3 + 1 ;
      }
      else {
          return (SA12[t] - n0) * 3 + 2;
      }
   }    
   i = GetI(); // pos of current offset 12 suffix
   j = SA0[p]; // pos of current offset 0 suffix
   var flag:bool;
   if (SA12[t] < n0) {
             // different compares for mod 1 and mod 2 suffixes
             // i % 3 =1
          flag=leq(s[i], s12[SA12[t] + n0], s[j], s12[j/3]); 
   } else {
             // i % 3 =2
          flag=leq(s[i],s[i+1],s12[SA12[t]-n0+1], s[j],s[j+1],s12[j/3+n0]);
    // flag=leq(s[i],s[i+1],s12[SA12[t]-n0], s[j],s[j+1],s12[j/3+n0]);
   }
   if (flag)
   {// suffix from SA12 is smaller
      SA[k] = i; 
      k=k+1;
      t=t+1;
      if (t == n02)  {// done --- only SA0 suffixes left
         forall (i1,j1) in zip (k..n-1,p..p+n-k-1) do  SA[i1] = SA0[j1];
         break;
      }  
   } else {// suffix from SA0 is smaller
         SA[k] = j; 
         k=k+1;
         p=p+1;
         var tmpt=t:int;
         if (p == n0) { // done --- only SA12 suffixes left
             for i1 in tmpt..n02-1 do { 
                           SA[k] = GetI();
                           t=t+1;
                           k=k+1;
             }
             break;
         }  
     }    
   }
}



}
