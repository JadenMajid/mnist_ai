#ifndef LINALG_H
#define LINALG_H

#include <cstdlib>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define true 1
#define false 0

typedef struct Mat {
  float *values;
  int n;
  int m;
} Mat;

Mat *new_mat(int n, int m) {
  Mat *new_mat = (Mat *)malloc(sizeof(Mat));
  new_mat->m = m;
  new_mat->n = n;
  new_mat->values = (float *)calloc(n * m, sizeof(float));
  return new_mat;
}

Mat *new_empty_mat() { return (Mat *)malloc(sizeof(Mat)); }

void delete_mat(Mat *mat) {
  free(mat->values);
  free(mat);
}

Mat *copy_mat(Mat *mat) {
  Mat *new_mat = (Mat *)malloc(sizeof(Mat));
  new_mat->m = mat->m;
  new_mat->n = mat->n;
  new_mat->values = (float *)calloc(mat->n * mat->m, sizeof(float));
  memcpy(new_mat->values, mat->values, sizeof(float) * mat->n * mat->m);
  return new_mat;
}

void print_mat(Mat *mat) {
  printf("\n[\n");
  for (int i = 0; i < mat->n; i++) {
    printf("[");
    for (int j = 0; j < mat->m; j++) {
      printf("%f, ", mat->values[i * mat->m + j]);
    }
    printf("\b\b]\n");
  }
  printf("]\n");
}

int save_mat(Mat *mat, char *name) {
  char path[1024];
  long int unix_time = (long int)time(NULL);

  sprintf(path, "./models/model%li", unix_time);
  // mkdir(path, 0777);
  // printf("\n%s",path);
  sprintf(path, "%s/%s", path, name);

  FILE *fd = (FILE *)fopen(path, "wb");
  if (fd == NULL) {
    printf("\nERROR OPENING FILE! %d\n", errno);
    return 1;
  }
  fwrite(mat, sizeof(int), 2, fd);
  fwrite(mat->values, sizeof(float), mat->n * mat->m, fd);
  fclose(fd);
  return 0;
}

Mat *load_mat(char *path) {
  Mat *mat = new_empty_mat();
  FILE *fd = fopen(path, "rb");
  if (fd == NULL) {
    printf("\nNO MODEL FOUND AT PATH!\n");
    return NULL;
  }
  fread(mat, sizeof(int), 2, fd);
  int len = mat->n * mat->m;
  mat->values = (float *)malloc(sizeof(float) * len);
  fread(mat->values, sizeof(float), mat->n * mat->m, fd);
  fclose(fd);
  return mat;
}

int equals(Mat *mat0, Mat *mat1) {
  if (mat0->n != mat1->n || mat0->m != mat1->m)
    return false;
  return !memcmp(mat0->values, mat1->values, sizeof(float) * mat0->n * mat0->m);
}

// out cannot be mat0 or mat1
void mat_dot_mat(Mat *mat0, Mat *mat1, Mat *out) {
  if (mat0->m != mat1->n | out->n != mat0->n | out->m != mat1->m) {
    printf("tried to dot mat of dim (%d,%d) with mat of dim (%d,%d)", mat0->n,
           mat0->m, mat1->n, mat1->m);
    abort();
  }
  if (out == NULL) {
    out = new_empty_mat();
  }
  // print_mat(mat0);
  // print_mat(mat1);
  float *out_current_values = out->values;
  float *temp = (float *)calloc(sizeof(float), mat0->n * mat1->m);

  for (int k = 0; k < mat0->n * mat1->m; k++) {
    for (int i = 0; i < mat0->m; i++) {
      // printf("%fx%f+", mat0->values[(k / mat1->n) * mat0->n + i],
      // mat1->values[(i * mat1->n + k % mat1->n)]);
      temp[k] += mat0->values[(k / mat1->m) * mat0->m + i] *
                 mat1->values[(i * mat1->m + k % mat1->m)];
    }
  }
  free(out->values);
  out->n = mat0->n;
  out->m = mat1->m;
  out->values = temp;
}
void mat_hamard_mat(Mat *mat0, Mat *mat1, Mat *out) {
  if (mat0->m != mat1->n | out->n != mat0->n | out->m != mat1->m) {
    printf("tried to hamard mat of dim (%d,%d) with mat of dim (%d,%d)",
           mat0->n, mat0->m, mat1->n, mat1->m);
    abort();
  }
  // print_mat(mat0);
  // print_mat(mat1);

  for (int k = 0; k < mat0->n * mat1->m; k++) {
    // printf("%fx%f+", mat0->values[(k / mat1->n) * mat0->n + i],
    // mat1->values[(i * mat1->n + k % mat1->n)]);
    out->values[k] = mat0->values[k] * mat1->values[k];
  }
}
void mat_T(Mat *mat) {

  float temp;

  float *new_vals = (float *)calloc(sizeof(float), mat->n * mat->m);
  for (int i = 0; i < mat->m; i++) {
    for (int j = 0; j < mat->n; j++) {
      new_vals[i * mat->n + j] = mat->values[j * mat->m + i];
    }
  }

  int temp_int;
  temp_int = mat->n;
  mat->n = mat->m;
  mat->m = temp_int;
  free(mat->values);
  mat->values = new_vals;
}

Mat *mat_add_mat(Mat *mat0, Mat *mat1, Mat *out) {
  if (mat0->n != mat1->n || mat0->m != mat1->m) {
    printf("\ntried to add two matricies with different dimensions!%dx%d+%dx%d",
           mat0->n, mat0->m, mat1->n, mat1->m);
    abort();
  }
  if (!out) {
    out = new_mat(mat0->m, mat1->n);
  }
  // regular addition of two matrices
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] + mat1->values[i];
  }
  return out;
}
Mat *mat_sub_mat(Mat *mat0, Mat *mat1, Mat *out) {
  if (mat0->n != mat1->n || mat0->m != mat1->m) {
    printf("\ntried to add two matricies with different dimensions!%dx%d+%dx%d",
           mat0->n, mat0->m, mat1->n, mat1->m);
    abort();
  }
  if (!out) {
    out = new_mat(mat0->m, mat1->n);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] - mat1->values[i];
  }
  return out;
}

Mat *mat_add_rvec(Mat *mat0, Mat *rvec, Mat *out) {
  if (rvec->n != 1 || mat0->m != rvec->m) {
    printf("\ntried to add mat to rowvec with different dimensions!%dx%d+%dx%d",
           mat0->n, mat0->m, rvec->n, rvec->m);
    abort();
  }
  if (!out) {
    out = new_mat(mat0->m, rvec->n);
  }
  // regular addition of two matrices
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] + rvec->values[i % mat0->m];
  }
  return out;
}

Mat *mat_add_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->m, mat0->n);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] + scalar;
  }
  return out;
}
Mat *mat_sub_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->m, mat0->n);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] - scalar;
  }
  return out;
}
Mat *mat_mul_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->m, mat0->n);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] * scalar;
  }
  return out;
}
Mat *mat_div_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->m, mat0->n);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] / scalar;
  }
  return out;
}

Mat *mat_apply_fn(Mat *mat0, float (*fn)(float), Mat *out) {
  if (!out) {
    out = new_mat(mat0->m, mat0->n);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = fn(mat0->values[i]);
  }
  return out;
}

Mat *mat_apply_fn(Mat *mat0, float (*fn)(), Mat *out) {
  if (!out) {
    out = new_mat(mat0->m, mat0->n);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = fn();
  }
  return out;
}

float rand_float() { return (float)rand(); }
Mat *randomize(Mat *mat0) { return mat_apply_fn(mat0, rand_float, mat0); }

#endif
