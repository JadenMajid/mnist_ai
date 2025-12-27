#include "linalg.h"

Mat *mat_new(int rows, int cols) {
    Mat *m = malloc(sizeof(Mat));
    m->rows = rows;
    m->cols = cols;
    m->data = calloc(rows * cols, sizeof(float));
    return m;
}

void mat_free(Mat *m) {
    if (m) {
        free(m->data);
        free(m);
    }
}

Mat *mat_copy(Mat *src) {
    Mat *dst = mat_new(src->rows, src->cols);
    memcpy(dst->data, src->data, sizeof(float) * src->rows * src->cols);
    return dst;
}

void mat_print(Mat *m) {
    printf("[%d x %d]\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%8.4f ", m->data[i * m->cols + j]);
        }
        printf("\n");
    }
}



// Optimized Dot Product (Row-Major Order)
void mat_dot(Mat *a, Mat *b, Mat *out) {
    if (a->cols != b->rows) {
        fprintf(stderr, "Error: Inner dimensions must match for dot product. (%dx%d)*(%dx%d)\n", a->rows, a->cols, b->rows, b->cols);
        return;
    }
    
    // Use a temp buffer if 'out' is one of the inputs to prevent data corruption
    float *res = calloc(a->rows * b->cols, sizeof(float));

    for (int i = 0; i < a->rows; i++) {
        for (int k = 0; k < a->cols; k++) { // Swapped loops for cache efficiency
            float va = a->data[i * a->cols + k];
            for (int j = 0; j < b->cols; j++) {
                res[i * b->cols + j] += va * b->data[k * b->cols + j];
            }
        }
    }
    
    free(out->data);
    out->rows = a->rows;
    out->cols = b->cols;
    out->data = res;
}

Mat *mat_dot_new(Mat *a, Mat *b){
    Mat *out = mat_new(a->rows, b->cols);
    mat_dot(a, b, out);
    return out;
}



void mat_add(Mat *a, Mat *b, Mat *out) {
    for (int i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void  mat_rand(Mat *m, float low, float high) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        float scale = (float)rand() / (float)RAND_MAX;
        m->data[i] = low + scale * (high - low);
    }
}

void mat_transpose(Mat *m, Mat *out){
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            out->data[j * m->rows + i] = m->data[i * m->cols + j];
        }
    }

}

Mat *mat_transpose_new(Mat *m) {
    Mat *out = mat_new(m->cols, m->rows);

    return out; // Ensure this line exists!
}

#include "linalg.h"

// Fills every element of the matrix with a constant value
void mat_fill(Mat *m, float val) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = val;
    }
}

// Subtracts matrix b from a: out = a - b
void mat_sub(Mat *a, Mat *b, Mat *out) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in mat_sub (%dx%d vs %dx%d)\n", 
                a->rows, a->cols, b->rows, b->cols);
        return;
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

// Multiplies every element by a scalar: out = m * s
void mat_scale(Mat *m, float s, Mat *out) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        out->data[i] = m->data[i] * s;
    }
}

// Element-wise multiplication (Schur product): out = a .* b
void mat_hadamard(Mat *a, Mat *b, Mat *out) {
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in mat_hadamard\n");
        return;
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

// Applies a function pointer to every element (useful for activations)
void mat_apply(Mat *m, float (*fn)(float), Mat *out) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        out->data[i] = fn(m->data[i]);
    }
}
/*
takes row or column vector, needs to have same non-zero dim as z->cols
*/
void mat_add_bias(Mat *z, Mat *b) {
    // z is (Batch x Features), b is (1 x Features)
    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {
            // Add the j-th bias component to the j-th feature of every row
            z->data[i * z->cols + j] += b->data[j];
        }
    }
}

void mat_softmax(Mat *m, Mat *out)
{
    for (int i = 0; i < m->rows; i++){
        // find max 
        float rowmax = __FLT_MIN__;
        for (int j = 0; j < m->cols;j++){
            if (m->data[i*m->cols+j] > rowmax) 
                rowmax = m->data[i*m->cols+j];
        }
        // assign out values
        float row_exp_sum = 0.0;
        for (int j = 0; j < m->cols;j++){
            out->data[i*m->cols+j] = expf(m->data[i*m->cols+j] - rowmax);
            row_exp_sum += out->data[i*m->cols+j];
        }
        // div by sum of exps
        for (int j = 0; j < m->cols;j++){
            out->data[i*m->cols+j] /= row_exp_sum;
        }
    }
}
