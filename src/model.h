#ifndef MODEL_H
#define MODEL_H
#include <linalg.h>
#include <ml_math.h>


typedef struct Model
{
    int n_layers;
    Layer *layers;
} Model;



typedef struct Layer
{
    Mat *w;
    Mat *b;
} Layer;

Mat *apply(Mat *activations, Layer *l)
{
    if (activations->n != l->w->m)
    {
        printf("\nACTIVATIONS n NOT EQUAL TO WEIGHT m!%ix%i*%ix%i", activations->m, activations->n, l->w->m, l->w->n);
    }
    mat_dot_mat(activations, l->w, activations);
    mat_add_mat(activations, l->b, activations);
    mat_apply_fn(activations, logistic, activations);
    return activations;
}

Mat *predict(Model *m, Mat *X)
{
    for (int i = 0; i < m->n_layers; i++)
    {
        X = apply(X, &m->layers[i]);
    }
}

#endif