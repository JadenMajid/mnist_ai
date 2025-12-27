#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "fs.h"

// Helper to handle path concatenation safely
void build_path(char *output, size_t size, const char *dir, const char *name) {
    snprintf(output, size, "%s/%s", dir, name);
}

int save_mat(char *dir, char *name, Mat *mat) {
    char path[512];
    build_path(path, sizeof(path), dir, name);

    FILE *fp = fopen(path, "wb"); // Use double quotes for string literals
    if (!fp) return -1;

    fwrite(&mat->rows, sizeof(int), 1, fp);
    fwrite(&mat->cols, sizeof(int), 1, fp);
    // Write data directly from the pointer, not the address of the pointer variable
    fwrite(mat->data, sizeof(float), mat->rows * mat->cols, fp);
    
    fclose(fp);
    return 0;
}

Mat *load_mat(char *dir, char *name) {
    char path[512];
    build_path(path, sizeof(path), dir, name);

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    int rows, cols;
    fread(&rows, sizeof(int), 1, fp);
    fread(&cols, sizeof(int), 1, fp);

    Mat *mat = mat_new(rows, cols);
    fread(mat->data, sizeof(float), rows * cols, fp);

    fclose(fp);
    return mat;
}

int save_layer(char *dir, int num, Layer *l) {
    char w_name[64], b_name[64];
    snprintf(w_name, sizeof(w_name), "layer_%d_W.bin", num);
    snprintf(b_name, sizeof(b_name), "layer_%d_B.bin", num);
    
    save_mat(dir, w_name, l->w); // Assuming l->w is Mat*
    save_mat(dir, b_name, l->b);
    return 0;
}

int save_model(char *dirname, Model *model) {
    char path[256];
    snprintf(path, sizeof(path), "data/%s", dirname);
    
    // Create directory (Linux specific - useful for your Arch setup)
    mkdir("data", 0777); 
    mkdir(path, 0777);

    char meta_path[512];
    build_path(meta_path, sizeof(meta_path), path, "model_info.bin");

    FILE *fp = fopen(meta_path, "wb");
    if (!fp) return -1;

    fwrite(&model->n_layers, sizeof(int), 1, fp);
    fwrite(&model->max_epochs, sizeof(int), 1, fp);
    fwrite(&model->lr, sizeof(float), 1, fp);
    fclose(fp);

    for (int i = 0; i < model->n_layers; i++) {
        save_layer(path, i, &model->layers[i]);
    }
    return 0;
}

Model *load_model(char *dirname) {
    char path[256];
    snprintf(path, sizeof(path), "data/%s", dirname);

    char meta_path[512];
    build_path(meta_path, sizeof(meta_path), path, "model_info.bin");

    FILE *fp = fopen(meta_path, "rb");
    if (!fp) return NULL;

    int n_layers, epochs;
    float lr;
    fread(&n_layers, sizeof(int), 1, fp);
    fread(&epochs, sizeof(int), 1, fp);
    fread(&lr, sizeof(float), 1, fp);
    fclose(fp);

    Model *model = malloc(sizeof(Model));
    model->n_layers = n_layers;
    model->max_epochs = epochs;
    model->lr = lr;
    model->layers = malloc(sizeof(Layer) * n_layers);

    for (int i = 0; i < n_layers; i++) {
        char w_name[64], b_name[64];
        snprintf(w_name, sizeof(w_name), "layer_%d_W.bin", i);
        snprintf(b_name, sizeof(b_name), "layer_%d_B.bin", i);
        
        model->layers[i].w = load_mat(path, w_name);
        model->layers[i].b = load_mat(path, b_name);
    }

    return model;
}