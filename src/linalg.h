#ifndef LINALG_H
#define LINALG_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#define true 1
#define false 0

typedef struct Mat
{
        int m;
    int n;
    float *values;
} Mat;

Mat *new_mat(int m, int n)
{
    Mat *new_mat = malloc(sizeof(Mat));
    new_mat->n = n;
    new_mat->m = m;
    new_mat->values = calloc(m * n, sizeof(float));
    return new_mat;
}

Mat *new_empty_mat()
{
    return malloc(sizeof(Mat));
}

int delete_mat(Mat *mat){
    free(mat->values);
    free(mat);
}

Mat *copy_mat(Mat *mat)
{
    Mat *new_mat = malloc(sizeof(Mat));
    new_mat->n = mat->n;
    new_mat->m = mat->m;
    new_mat->values = calloc(mat->m * mat->n, sizeof(float));
    memcpy(new_mat->values, mat->values, sizeof(float) * mat->m * mat->n);
    // for (int i = 0; i < new_mat->m; i++)
    // {
    //     for (int j = 0; j < new_mat->n; j++)
    //     {
    //         new_mat->values[new_mat->n * i + j] = mat->values[new_mat->n * i + j];
    //     }
    // }
    return new_mat;
}

void print_mat(Mat *mat)
{
    // for(int i = 0; i < mat->m*mat->n; i++){
    //     printf("%f\n", mat->values[i]);
    // }
    printf("\n[\n");
    for (int i = 0; i < mat->m; i++)
    {
        printf("[");
        for (int j = 0; j < mat->n; j++)
        {
            printf("%f, ", mat->values[i * mat->n + j]);
        }
        printf("\b\b]\n");
    }
    printf("]\n");
}

int save_mat(Mat *mat, char *name)
{
    char path[1024];
    // printf("JFDSLKJFKLDSJKLFDSJDSKLJFKLDSJLKFJDSJ");

    long int unix_time = (long int)time(NULL);

    sprintf(path, "./models/model%li", unix_time);
    mkdir(path, 0777);
    // printf("\n%s",path);
    sprintf(path, "%s/%s", path, name);

    FILE *fd = (FILE *)fopen(path, "wb");
    if (fd == NULL)
    {
        printf("\nERROR OPENING FILE! %d\n", errno);
        return 1;
    }
    fwrite(mat, sizeof(int), 2, fd);
    fwrite(mat->values, sizeof(float), mat->m * mat->n, fd);
    fclose(fd);
    return 0;
}

Mat *load_mat(char *path)
{
    Mat *mat = new_empty_mat();
    FILE *fd = fopen(path, "rb");
    if (fd == NULL)
    {
        printf("\nNO MODEL FOUND AT PATH!\n");
        return NULL;
    }
    fread(mat, sizeof(int), 2, fd);
    int len = mat->m * mat->n;
    mat->values = malloc(sizeof(float) * len);
    fread(mat->values, sizeof(float), mat->m * mat->n, fd);
    fclose(fd);
    return mat;
}

int equals(Mat *mat0, Mat *mat1)
{
    if (mat0->m != mat1->m || mat0->n != mat1->n)
        return false;
    // for(int i =0;i<mat0->m*mat0->n;i++){
    //     if (mat0->values[i]-mat1->values[i]>0.0001){
    //         return false;
    //     }
    // }
    // return true;
    return !memcmp(mat0->values, mat1->values, sizeof(float) * mat0->m * mat0->n);
}

// out cannot be mat0 or mat1
void mat_dot_mat(Mat *mat0, Mat *mat1, Mat *out)
{
    if (mat0->n != mat1->m | out->m != mat0->m | out->n != mat1->n)
    {
        printf("tried to dot mat of dim (%d,%d) with mat of dim (%d,%d)", mat0->m, mat0->n, mat1->m, mat1->n);
        abort();
    }
    // print_mat(mat0);
    // print_mat(mat1);
    float *out_current_values = out->values;
    float *temp = calloc(sizeof(float), mat0->m * mat1->n);

    for (int k = 0; k < mat0->m * mat1->n; k++)
    {
        for (int i = 0; i < mat0->n; i++)
        {
            // printf("%fx%f+", mat0->values[(k / mat1->n) * mat0->n + i], mat1->values[(i * mat1->n + k % mat1->n)]);
            temp[k] += mat0->values[(k / mat1->n) * mat0->n + i] * mat1->values[(i * mat1->n + k % mat1->n)];
        }
    }
    free(out->values);
    out->m = mat0->m;
    out->n = mat1->n;
    out->values = temp;
}
void mat_hamard_mat(Mat *mat0, Mat *mat1, Mat *out)
{
    if (mat0->n != mat1->m | out->m != mat0->m | out->n != mat1->n)
    {
        printf("tried to hamard mat of dim (%d,%d) with mat of dim (%d,%d)", mat0->m, mat0->n, mat1->m, mat1->n);
        abort();
    }
    // print_mat(mat0);
    // print_mat(mat1);

    for (int k = 0; k < mat0->m * mat1->n; k++)
    {
            // printf("%fx%f+", mat0->values[(k / mat1->n) * mat0->n + i], mat1->values[(i * mat1->n + k % mat1->n)]);
            out->values[k] = mat0->values[k]*mat1->values[k]; 
    }
}
void mat_T(Mat *mat)
{

    float temp;

    float *new_vals = calloc(sizeof(float), mat->m * mat->n);
    for (int i = 0; i < mat->n; i++)
    {
        for (int j = 0; j < mat->m; j++)
        {
            new_vals[i * mat->m + j] = mat->values[j * mat->n + i];
        }
    }

    int temp_int;
    temp_int = mat->m;
    mat->m = mat->n;
    mat->n = temp_int;
    free(mat->values);
    mat->values = new_vals;
}

void mat_add_mat(Mat *mat0, Mat *mat1, Mat *out)
{
    if (mat0->m != mat1->m || mat0->n != mat1->n)
    {
        printf("\ntried to add two matricies with different dimensions!%dx%d+%dx%d", mat0->m, mat0->n, mat1->m, mat1->n);
        abort();
    }
    for (int i = 0; i < mat0->m * mat0->n; i++)
    {
        out->values[i] = mat0->values[i] + mat1->values[i];
    }
}
void mat_sub_mat(Mat *mat0, Mat *mat1, Mat *out)
{
    if (mat0->m != mat1->m || mat0->n != mat1->n)
    {
        printf("\ntried to add two matricies with different dimensions!%dx%d+%dx%d", mat0->m, mat0->n, mat1->m, mat1->n);
        abort();
    }
    for (int i = 0; i < mat0->m * mat0->n; i++)
    {
        out->values[i] = mat0->values[i] - mat1->values[i];
    }
}

void mat_add_scalar(Mat *mat0, float scalar, Mat *out)
{
    for (int i = 0; i < mat0->m * mat0->n; i++)
    {
        out->values[i] = mat0->values[i] + scalar;
    }
}
void mat_sub_scalar(Mat *mat0, float scalar, Mat *out)
{
    for (int i = 0; i < mat0->m * mat0->n; i++)
    {
        out->values[i] = mat0->values[i] - scalar;
    }
}
void mat_mul_scalar(Mat *mat0, float scalar, Mat *out)
{
    for (int i = 0; i < mat0->m * mat0->n; i++)
    {
        out->values[i] = mat0->values[i] * scalar;
    }
}
void mat_div_scalar(Mat *mat0, float scalar, Mat *out)
{
    for (int i = 0; i < mat0->m * mat0->n; i++)
    {
        out->values[i] = mat0->values[i] / scalar;
    }
}
void mat_apply_fn(Mat *mat0, float (*fn)(float), Mat *out)
{
    for (int i = 0; i < mat0->m * mat0->n; i++)
    {
        out->values[i] = fn(mat0->values[i]);
    }
}
#endif