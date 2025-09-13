static int GlobalTotalTests;
static int GlobalFailedTests;

#include "../src/linalg.h"

#define AssertTrue(Expression) \
++GlobalTotalTests; \
if(!(Expression)) \
{ \
    ++GlobalFailedTests; \
    printf("%s(%d): expression assert fail.\n", __FILE__, __LINE__); \
}

int main(int ArgCount, char *Args[]) {
    // Do some tests
    struct Vec* test_vec = new_vec(1);
    AssertTrue(test_vec->len == 1);
    AssertTrue(test_vec->values[0]==0);
    vec_add_scalar(test_vec, 1,test_vec);
    AssertTrue(test_vec->values[0]==1);
    vec_mul_scalar(test_vec, 2, test_vec);
    AssertTrue(test_vec->values[0]==2);
    free(test_vec);

    test_vec = new_vec(3);
    for (int i = 0;i<3;i++) test_vec->values[i]=i;
    AssertTrue(vec_dot_vec(test_vec,test_vec)==5);

    struct Mat* test_mat = new_mat(3,3);
    for (int i = 0;i<9;i++) test_mat->values[i]=i;

    struct Vec* out_vec = new_vec(3);
    mat_dot_vec(test_mat, test_vec, out_vec);
    float *result = malloc(sizeof(float)*3);
    result[0]=5.;
    result[1]=14.;
    result[2]=23.;

    for (int i = 0;i<3;i++) {
        AssertTrue(result[i]==out_vec->values[i]);
    }
    free(result);

    struct Mat* result_mat = new_mat(3,3);
    mat_dot_mat(test_mat, test_mat, result_mat);
    printf("[");
    for (int i = 0;i<3;i++) {
        printf("[");
        for (int j = 0;j<3;j++) {
            printf("%f,",result_mat->values[i*3+j]);
        }
        printf("],\n");
    }
    printf("]\n");
    


    int Result = (GlobalFailedTests != 0);

    printf("Unit Tests %s: %d/%d passed.\n",
           Result ? "Failed" : "Successful", 
           GlobalTotalTests - GlobalFailedTests, 
           GlobalTotalTests);  
}

