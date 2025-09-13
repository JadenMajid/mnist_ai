#ifndef LINALG_H
#define LINALG_H

#include <stdlib.h>
#include <stdio.h>

struct Vec{
    int len;
    float* values;
};

struct Mat{
    int m;
    int n;
    float* values;
};

struct Mat* new_mat(int n, int m){
    struct Mat* new_mat = malloc(sizeof(struct Mat));
    new_mat->n = n;
    new_mat->m = m;
    new_mat->values = calloc(m*n,sizeof(float));
    return new_mat;
}

struct Mat* copy_mat(struct Mat* mat){
    struct Mat* new_mat = malloc(sizeof(struct Mat));
    new_mat->n = mat->n;
    new_mat->m = mat->m;
    new_mat->values = calloc(mat->m*mat->n,sizeof(float));
    for(int i =0;i<new_mat->m;i++){
        for(int j =0;j<new_mat->n;j++){
            new_mat->values[new_mat->m*i+j] = mat->values[new_mat->m*i+j];
        }
    }
    return new_mat;
}

void mat_dot_vec(struct Mat* mat, struct Vec* vec, struct Vec* out){
    if (vec->len!=mat->n||mat->m!= out->len){
        printf("tried to dot mat of dim (%d,%d) with vec of len %d", mat->m, mat->n, vec->len);
        abort();
    }
    float temp;
    for (int i = 0;i<mat->m;i++){
        temp = 0.0;
        for (int j =0;j<mat->n;j++){
            //printf("\n%fx%f",mat->values[mat->m*i+j],vec->values[i]);
            temp += mat->values[mat->m*i+j]*vec->values[j];
        }
        //printf("\n%f,",temp);
        out->values[i]=temp;
    }
}
// out cannot be mat0 or mat1
void mat_dot_mat(struct Mat* mat0, struct Mat* mat1, struct Mat* out){
    if (mat0->n!=mat1->m|out->n!=mat0->m|out->m!=mat1->n){
        printf("tried to dot mat of dim (%d,%d) with mat of dim (%d,%d)", mat0->m, mat0->n, mat1->m, mat1->n);
        abort();
    }
    float temp;
    for(int k = 0;k<mat0->n*mat1->m;k++){
        temp = 0.0;

        for (int i = 0;i<mat0->n;i++){
            printf("%fx%f+",mat0->values[(k/mat0->n)*mat0->n+i] , mat1->values[(i*mat1->m+k%mat1->m)]);
            temp += mat0->values[(k/mat0->n)*mat0->n+i] * mat1->values[(i*mat1->m + k%mat1->m)];
        }
        printf("\n");
        out->values[k]=temp;
    }
}

struct Vec* new_vec(int len){
    struct Vec* new_vec = malloc(sizeof(struct Vec));
    new_vec->len = len;
    new_vec->values = calloc(len,sizeof(float));
    return new_vec;
}
struct Vec* copy_vec(struct Vec* vec){
    struct Vec* new_vec = malloc(sizeof(struct Vec));
    new_vec->len = vec->len;
    new_vec->values = calloc(vec->len,sizeof(float));
    for (int i = 0;i<vec->len;i++){
        new_vec->values[i] = vec->values[i];
    }
    return new_vec;
}
void vec_add_scalar(struct Vec* vec, float scalar, struct Vec* out){
    for (int i = 0;i<vec->len;i++){
        out->values[i] = vec->values[i] + scalar;
    }
}
void vec_mul_scalar(struct Vec* vec, float scalar, struct Vec* out){
    for (int i = 0;i<vec->len;i++){
        out->values[i] = vec->values[i] * scalar;
    }
}
void vec_add_vec(struct Vec* vec0, struct Vec* vec1, struct Vec* out){
    if (vec0->len!=vec1->len||vec0->len != out->len){
        printf("tried to add vec of len %i with vec of len %i", vec0->len, vec1->len);
        abort();
    }
    for (int i = 0;i<vec0->len;i++){
        out->values[i] = vec0->values[i] + vec1->values[i];
    }
}

int vec_dot_vec(struct Vec* vec0, struct Vec* vec1){
    if (vec0->len!=vec1->len){
        printf("tried to dot vec of len %i with vec of len %d", vec0->len, vec1->len);
        abort();
    }
    float out = 0.0;
    for (int i = 0;i<vec0->len;i++){
        out += vec0->values[i] * vec1->values[i];
    }
    return out;
}



#endif