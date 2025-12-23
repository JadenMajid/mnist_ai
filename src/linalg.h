#ifndef LINALG_H
#define LINALG_H

#include <stdlib.h>
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
  struct Mat *T;
} Mat;

Mat *new_mat(int n, int m) {
  Mat *new_mat = (Mat *)malloc(sizeof(Mat));
  new_mat->m = m;
  new_mat->n = n;
  new_mat->values = (float *)calloc(n * m, sizeof(float));
  new_mat->T = (Mat *)malloc(sizeof(Mat));
  new_mat->T->n = m;
  new_mat->T->m = n;
  new_mat->T->values = (float *)calloc(m * n, sizeof(float));
  new_mat->T->T = new_mat;
  return new_mat;
}

Mat *new_empty_mat() { return (Mat *)malloc(sizeof(Mat)); }

void delete_mat(Mat *mat) {
  if (!mat) return;
  if (mat->T) {
    Mat *t = mat->T;
    mat->T = NULL;
    if (t->T == mat) t->T = NULL;
    free(t->values);
    free(t);
  }
  if (mat->values) free(mat->values);
  free(mat);
}

Mat *copy_mat(Mat *mat) {
  Mat *res = new_mat(mat->n, mat->m);
  memcpy(res->values, mat->values, sizeof(float) * mat->n * mat->m);
  // populate transpose
  for (int i = 0; i < mat->n; i++) {
    for (int j = 0; j < mat->m; j++) {
      res->T->values[j * res->n + i] = res->values[i * res->m + j];
    }
  }
  return res;
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
  fwrite(&(mat->n), sizeof(int), 1, fd);
  fwrite(&(mat->m), sizeof(int), 1, fd);
  fwrite(mat->values, sizeof(float), mat->n * mat->m, fd);
  fclose(fd);
  return 0;
}

Mat *load_mat(char *path) {
  Mat *mat = new_empty_mat();
  mat->values = NULL;
  mat->T = NULL;
  FILE *fd = fopen(path, "rb");
  if (fd == NULL) {
    printf("\nNO MODEL FOUND AT PATH!\n");
    return NULL;
  }
  int n, m;
  fread(&n, sizeof(int), 1, fd);
  fread(&m, sizeof(int), 1, fd);
  mat->n = n;
  mat->m = m;
  int len = n * m;
  mat->values = (float *)malloc(sizeof(float) * len);
  fread(mat->values, sizeof(float), len, fd);
  // create transpose
  mat->T = (Mat *)malloc(sizeof(Mat));
  mat->T->n = mat->m;
  mat->T->m = mat->n;
  mat->T->values = (float *)malloc(sizeof(float) * len);
  mat->T->T = mat;
  for (int i = 0; i < mat->n; i++) {
    for (int j = 0; j < mat->m; j++) {
      mat->T->values[j * mat->n + i] = mat->values[i * mat->m + j];
    }
  }
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
  float *temp = (float *)calloc(mat0->n * mat1->m, sizeof(float));

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
  // ensure transpose exists and is correct
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
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
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
}
void mat_T(Mat *mat) {

  float temp;

  if (!mat->T) {
    // create transpose if missing
    mat->T = (Mat *)malloc(sizeof(Mat));
    mat->T->n = mat->m;
    mat->T->m = mat->n;
    mat->T->values = (float *)calloc(mat->m * mat->n, sizeof(float));
    mat->T->T = mat;
  }
  // populate transpose buffer from current values
  for (int i = 0; i < mat->n; i++) {
    for (int j = 0; j < mat->m; j++) {
      mat->T->values[j * mat->n + i] = mat->values[i * mat->m + j];
    }
  }
  // swap values pointers and dimensions with transpose
  Mat *t = mat->T;
  float *tmp_vals = mat->values;
  mat->values = t->values;
  t->values = tmp_vals;
  int tmp_n = mat->n;
  int tmp_m = mat->m;
  mat->n = t->n;
  mat->m = t->m;
  t->n = tmp_n;
  t->m = tmp_m;
}

Mat *mat_add_mat(Mat *mat0, Mat *mat1, Mat *out) {
  if (mat0->n != mat1->n || mat0->m != mat1->m) {
    printf("\ntried to add two matricies with different dimensions!%dx%d+%dx%d",
           mat0->n, mat0->m, mat1->n, mat1->m);
    abort();
  }
  if (!out) {
    out = new_mat(mat0->n, mat0->m);
  }
  // regular addition of two matrices
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] + mat1->values[i];
  }
  // update transpose
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
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
    out = new_mat(mat0->n, mat0->m);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] - mat1->values[i];
  }
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
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
    out = new_mat(mat0->n, mat0->m);
  }
  // regular addition of two matrices
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] + rvec->values[i % mat0->m];
  }
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
  return out;
}

Mat *mat_add_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->n, mat0->m);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] + scalar;
  }
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
  return out;
}
Mat *mat_sub_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->n, mat0->m);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] - scalar;
  }
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
  return out;
}
Mat *mat_mul_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->n, mat0->m);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] * scalar;
  }
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
  return out;
}
Mat *mat_div_scalar(Mat *mat0, float scalar, Mat *out) {
  if (!out) {
    out = new_mat(mat0->n, mat0->m);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = mat0->values[i] / scalar;
  }
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
  return out;
}

Mat *mat_apply_fn(Mat *mat0, float (*fn)(float), Mat *out) {
  if (!out) {
    out = new_mat(mat0->n, mat0->m);
  }
  for (int i = 0; i < mat0->n * mat0->m; i++) {
    out->values[i] = fn(mat0->values[i]);
  }
  if (!out->T) {
    out->T = (Mat *)malloc(sizeof(Mat));
    out->T->n = out->m;
    out->T->m = out->n;
    out->T->values = (float *)calloc(out->n * out->m, sizeof(float));
    out->T->T = out;
  }
  for (int i = 0; i < out->n; i++) {
    for (int j = 0; j < out->m; j++) {
      out->T->values[j * out->n + i] = out->values[i * out->m + j];
    }
  }
  return out;
}

float rand_float(float _) { (void)_; return (float)rand(); }
Mat *randomize(Mat *mat0) { return mat_apply_fn(mat0, rand_float, mat0); }

float norm_L2_squared(Mat *mat){
  if (mat->n != 0 && mat->m != 0){
    printf("\ntried to get norm of matrix with bad dims!%dx%d",mat->n, mat->m);
    abort();
  }
  float sum = 0.0;
  Mat *column_vec = mat;
  if (mat->n == 1) column_vec = mat->T;
  for (int i = 0; i < column_vec->n;i++){
    sum += column_vec->values[i];
  }
  return sum;
}

#endif
