#ifndef FILES_H
#define FILES_H
#include "linalg.h"
#include "model.h"
#include <stdio.h>

#define OK 0
#define ERROR 1


int save_mat(char* dir, char* name, Mat *mat);
int save_model(char* name, Model *model);

Mat *load_mat(char *dir, char *name);
Model *load_model(char *dirname);

#endif