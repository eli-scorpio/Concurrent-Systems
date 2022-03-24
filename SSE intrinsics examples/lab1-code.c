//
// CSU33014 Lab 1
//

// Please examine version each of the following routines with names
// starting lab1. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics.

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".


#include <immintrin.h>
#include <stdio.h>

#include "lab1-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void lab1_routine0(float * restrict a, float * restrict b,
		    float * restrict c) {
  for (int i = 0; i < 1024; i++ ) {
    a[i] = b[i] * c[i];
  }
}

// here is a vectorized solution for the example above
void lab1_vectorized0(float * restrict a, float * restrict b,
		    float * restrict c) {
  __m128 a4, b4, c4;
  
  for (int i = 0; i < 1024; i = i+4 ) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    a4 = _mm_mul_ps(b4, c4);
    _mm_storeu_ps(&a[i], a4);
  }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float lab1_routine1(float * restrict a, float * restrict b,
		     int size) {
  float sum = 0.0;
  
  for ( int i = 0; i < size; i++ ) {
    sum = sum + a[i] * b[i];
  }
  return sum;
}

// insert vectorized code for routine1 here
float lab1_vectorized1(float * restrict a, float * restrict b,
		     int size) {
  // replace the following code with vectorized code
  __m128 sum4 = _mm_setzero_ps();
  __m128 a4, b4, prod4;

  int remainder = size%4;
  int i = 0;
  for(; i < size-remainder; i += 4 ) {
    a4 = _mm_loadu_ps(&a[i]);
    b4 = _mm_loadu_ps(&b[i]);
    prod4 = _mm_mul_ps(a4,b4);
    sum4 = _mm_add_ps(sum4, prod4);
  }

  float temp[4];
  _mm_storeu_ps(temp, sum4);
  float sum = temp[0]+temp[1]+temp[2]+temp[3];

  for(; i < size; i++)
    sum += a[i]*b[i];

  return sum;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void lab1_routine2(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = 1 - (1.0/(b[i]+1.0));
  }
}

// in the following, size can have any positive value
void lab1_vectorized2(float * restrict a, float * restrict b, int size) {
  __m128 vec1 = _mm_set1_ps(1.0);
  __m128 b4, addtemp, divtemp, subtemp;
  int remainder = size%4;
  int i;

  // replace the following code with vectorized code
  for (i = 0; i < size-remainder; i += 4) {
    a[i] = 1 - (1.0/(b[i]+1.0));
    b4 = _mm_loadu_ps(&b[i]);
    addtemp = _mm_add_ps(b4, vec1);
    divtemp = _mm_div_ps(vec1, addtemp);
    subtemp = _mm_sub_ps(vec1, divtemp);
    _mm_storeu_ps(&a[i], subtemp);
  }

  for(; i < size; i++)
    a[i] = 1 - (1.0/(b[i]+1.0));

}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void lab1_routine3(float * restrict a, float * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    if ( a[i] < 0.0 ) {
      a[i] = b[i];
    }
  }
}

// in the following, size can have any positive value
void lab1_vectorized3(float * restrict a, float * restrict b, int size) {
  __m128 a4, b4, mask;
  __m128 zero = _mm_set1_ps(0.0);

  int remainder = size%4;
  int i;

  // replace the following code with vectorized code
  for (i = 0; i < size-remainder; i += 4) {
    a4 = _mm_loadu_ps(&a[i]);
    mask = _mm_cmplt_ps(a4, zero);

    int newMask = _mm_movemask_ps(mask);

    if (newMask == 0xF) {
      b4 = _mm_loadu_ps(&b[i]);
      _mm_storeu_ps(&a[i], b4);
    }
    else if (newMask != 0){
      for(int j = 0; j < 4; j++)
        if(a[i+j] < 0.0)
          a[i+j] = b[i+j];
    }

  }

  for(; i < size; i++)
    if(a[i] < 0.0)
      a[i] = b[i];

}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void lab1_routine4(float * restrict a, float * restrict b,
		       float * restrict c) {
  for ( int i = 0; i < 2048; i = i+2  ) {
    a[i] = b[i]*c[i] - b[i+1]*c[i+1];
    a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];
  }
}

void lab1_vectorized4(float * restrict a, float * restrict b,
		       float * restrict  c) {
  // replace the following code with vectorized code
  __m128 b4, c4, product4, shuffle4, result1, result2, result;
  __m128 mask1 = _mm_set_ps(0,1,0,1); // 1010 backwards
  __m128 mask2 = _mm_set_ps(1,0,1,0); // 0101 backwards

  for ( int i = 0; i < 2048; i += 4) {
    b4 = _mm_loadu_ps(&b[i]);
    c4 = _mm_loadu_ps(&c[i]);
    product4 = _mm_mul_ps(b4, c4); // b0*c0 | b1*c1 | b2*c2 | b3*c3
    shuffle4 = _mm_shuffle_ps(product4, product4, _MM_SHUFFLE(2,3,0,1)); // b1*c1 | b0*c0 | b3*c3 | b2*c2
    result1 = _mm_sub_ps(product4, shuffle4);
    result1 = _mm_mul_ps(result1, mask1);
    //a[i] = b[i]*c[i] - b[i+1]*c[i+1];

    shuffle4 = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(2, 3, 0, 1));
    product4 = _mm_mul_ps(shuffle4, b4); // |b0*c1|b1*c0|b2*c3|b3*c2|
    shuffle4 = _mm_shuffle_ps(product4, product4, _MM_SHUFFLE(2,3,0,1)); // |c0*b1|c1*b0|c2*b3|c3*b2|
    result2 = _mm_add_ps(product4, shuffle4);
    result2 = _mm_mul_ps(result2, mask2);
    result = _mm_add_ps(result1, result2);
    _mm_storeu_ps(&a[i], result);
    //a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];
  }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void lab1_routine5(unsigned char * restrict a,
		    unsigned char * restrict b, int size) {
  for ( int i = 0; i < size; i++ ) {
    a[i] = b[i];
  }
}

void lab1_vectorized5(unsigned char * restrict a,
		       unsigned char * restrict b, int size) {
  __m128 b4;

  int remainder = size%4;
  int i;
  // replace the following code with vectorized code
  for (i = 0; i < size-remainder; i += 4){
    b4 = _mm_loadu_ps((float *) &b[i]);
    _mm_storeu_ps((float *) &a[i], b4);
  }

  for(; i < size; i++)
    a[i] = b[i];
}

/********************* routine 6 ***********************/

void lab1_routine6(float * restrict a, float * restrict b,
		       float * restrict c) {
  a[0] = 0.0;
  for ( int i = 1; i < 1023; i++ ) {
    float sum = 0.0;
    for ( int j = 0; j < 3; j++ ) {
      sum = sum +  b[i+j-1] * c[j];
    }
    a[i] = sum;
  }
  a[1023] = 0.0;
}

void lab1_vectorized6(float * restrict a, float * restrict b,
		       float * restrict c) {
  // replace the following code with vectorized code
  a[0] = 0.0;
  int size = 1023;
  int remainder = size%4;
  int i;

  __m128 c4 = _mm_set_ps(0, c[2], c[1], c[0]);
  __m128 sum1, sum2, sum4;
  sum1 = _mm_setzero_ps();
  sum2 = _mm_setzero_ps();
  sum4 = _mm_setzero_ps();

  for (i = 1; i < size-remainder; i += 4 ) {

    // load 4 vectors 
    __m128 bminus1 = _mm_loadu_ps(&b[i-1]); // |b[i-1]|b[i]|b[i+1]|b[i+2]
    __m128 b0 = _mm_loadu_ps(&b[i]); 
    __m128 b1 = _mm_loadu_ps(&b[i+1]);
    __m128 b2 = _mm_loadu_ps(&b[i+2]);

    // get product of vectors                                b-1[0]       b-1[1]      b-1[2]     b-1[3]
    __m128 productbminus1 = _mm_mul_ps(bminus1, c4); // | b[i-1]*c[0] | b[i]*c[1] | b[i+1]*c[2] | 0 |
    __m128 productb = _mm_mul_ps(b0, c4);
    __m128 productb1 = _mm_mul_ps(b1, c4);
    __m128 productb2 = _mm_mul_ps(b2, c4);

    // get sum
    sum1 = _mm_hadd_ps(productbminus1, productb); // | b-1[0]+b-1[1] | b-1[2]+b-1[3] | b[0]+b[1] | b[2]+b[3]
    sum2 = _mm_hadd_ps(productb1, productb2);
    sum4 = _mm_hadd_ps(sum1, sum2); // |b-1 sum| b sum | b1 sum | b2 sum |

    _mm_storeu_ps(&a[i], sum4);
  }


  for (; i < size; i++){
    float sum = 0.0;

    for ( int j = 0; j < 3; j++ )
      sum = sum +  b[i+j-1] * c[j];
    
    a[i] = sum;
  }

  a[1023] = 0.0;
}



