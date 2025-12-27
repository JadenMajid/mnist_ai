#ifndef LINALG_H
#define LINALG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int rows;
    int cols;
    float *data;
} Mat;

// Lifecycle
Mat *mat_new(int rows, int cols);
void mat_free(Mat *m);
Mat *mat_copy(Mat *src);

// Operations (Standard)
void mat_fill(Mat *m, float val);
void mat_rand(Mat *m, float low, float high);
void mat_print(Mat *m);

// Arithmetic (Out can be one of the inputs unless specified)
void mat_add(Mat *a, Mat *b, Mat *out);
void mat_sub(Mat *a, Mat *b, Mat *out);
void mat_scale(Mat *m, float s, Mat *out);
void mat_hadamard(Mat *a, Mat *b, Mat *out);

// Linear Algebra
void mat_dot(Mat *a, Mat *b, Mat *out); // Matrix Multiplication
Mat *mat_dot_new(Mat *a, Mat *b);
Mat *mat_transpose_new(Mat *m);             // Returns a NEW matrix
void mat_transpose(Mat *m, Mat *out);
void mat_apply(Mat *m, float (*fn)(float), Mat *out);
void mat_add_bias(Mat *z, Mat *b);

#endif