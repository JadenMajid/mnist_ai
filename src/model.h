#ifndef MODEL_H
#define MODEL_H
#include "linalg.h"
#include "ml_math.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

/*
 * Activation Functions
 */
float sigmoid(float x) { return 1 / (1 + pow(e, -x)); }

float sigmoid_d(float x) {
  float s = sigmoid(x);
  return s * (1 - s);
}
typedef struct Layer {
  Mat *w;
  Mat *b; // row vec
  float (*h)(float);
  float (*dh)(float);
} Layer;

typedef struct Model {
  int n_layers;
  Layer *layers;
} Model;

typedef struct ForwardPass {
  Mat *a;
  Mat *z;
} ForwardPass;

typedef struct BackwardsPass {
  Mat *d;
  Layer *next_layer;
} BackwardsPass;

Layer *new_layer(int n, int m) {
  Layer *layer = (Layer *)malloc(sizeof(Layer));
  layer->b = new_mat(1, m);
  layer->w = new_mat(n, m);
  return layer;
}

Model *new_model(Mat *layers, float (*activation_function)(float)) {
  if (layers->m != 1) {
    printf("passed layers matrix must be a column matrix!");
    print_mat(layers);
  }
  Model *model = (Model *)malloc(sizeof(Model));
  model->n_layers = layers->n;
  model->layers = (Layer *)calloc(sizeof(Layer *), model->n_layers);
  for (int i = 1; i < layers->n; i++) {
    Layer *layer = model->layers + i;
    int n = layers[i - 1].n;
    int m = layers[i].m;
    layer->b = new_mat(1, m);
    layer->w = new_mat(n, m);
  }
  return model;
}

Layer *randomize_layer(Layer *layer) {
  layer->b = randomize(layer->b);
  layer->w = randomize(layer->w);
  return layer;
}

Model *randomize_model(Model *model) {
  for (int i = 0; i < model->n_layers; i++) {
    randomize_layer(model->layers + i);
  }
}

int input_dim_model(Model *model) { return model->layers[0].w->n; }

Mat *apply(Layer *layer, Mat *in, Mat *out) {
  if (!out) {
    out = new_mat(in->n, layer->w->m);
  }
  mat_dot_mat(in, layer->w, out);
  mat_add_rvec(out, layer->b, out);
  return out;
}

Mat *predict(Model *model, Mat *X, Mat *y_hat) {
  Mat *z = NULL;
  Mat *h_z = X;
  for (int i = 0; i < model->n_layers; i++) {
    z = apply(&model->layers[i], h_z, NULL);
    h_z = mat_apply_fn(z, sigmoid, NULL);
  }
  return h_z;
}

ForwardPass *forward_pass(Layer *layer, Mat *a, ForwardPass *out) {
  out->a = new_mat(a->n, layer->w->m);
  out->z = new_mat(a->n, layer->w->m);

  mat_dot_mat(layer->w, a, out->a);
  mat_apply_fn(out->a, layer->h, out->z);
  return out;
}

BackwardsPass *backwards_pass_hidden(Layer *layer, ForwardPass *fp,
                                     BackwardsPass *bp, BackwardsPass *out) {
  out->d = new_mat(layer->w->m, 1);
  mat_dot_mat(bp->next_layer->w.T, bp->d, out->d);
}

#endif
