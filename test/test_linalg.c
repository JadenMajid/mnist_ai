static int GlobalTotalTests;
static int GlobalFailedTests;

#include "../src/linalg.h"

#define AssertTrue(Expression)                                           \
    ++GlobalTotalTests;                                                  \
    if (!(Expression))                                                   \
    {                                                                    \
        ++GlobalFailedTests;                                             \
        printf("%s(%d): expression assert fail.\n", __FILE__, __LINE__); \
    }

void test_3x3mul3x3()
{
    printf("testing 3x3*3x3 matmul\n");
    Mat *expected_mat = new_mat(3, 3);
    float expected_0[] = {15., 18., 21., 42., 54., 66., 69., 90., 111.};
    free(expected_mat->values);
    expected_mat->values = (float *)&expected_0;

    Mat *test_mat = new_mat(3, 3);
    for (int i = 0; i < 9; i++)
        test_mat->values[i] = i;

    Mat *result_mat = new_mat(3, 3);
    mat_dot_mat(test_mat, test_mat, result_mat);

    AssertTrue(equals(result_mat, expected_mat));

    delete_mat(expected_mat);
    delete_mat(test_mat);
    delete_mat(result_mat);
}

void test_3x3mul3x2()
{
    printf("testing 3x3*3x2 matmul\n");
    Mat *expected_mat = new_mat(3, 2);
    float expected[] = {10., 13.,
                        28., 40.,
                        46., 67.};
    free(expected_mat->values);
    expected_mat->values = (float *)&expected;

    Mat *test_mat0 = new_mat(3, 3);
    for (int i = 0; i < 9; i++)
        test_mat0->values[i] = i;
    Mat *test_mat1 = new_mat(3, 2);
    free(test_mat1->values);
    float test_mat1_values[] = {0., 1., 2., 3., 4., 5.};
    test_mat1->values = (float *)&test_mat1_values;

    Mat *result_mat = new_mat(3, 2);
    mat_dot_mat(test_mat0, test_mat1, result_mat);
    // print_mat(result_mat);
    AssertTrue(equals(result_mat, expected_mat));

    delete_mat(expected_mat);
    delete_mat(test_mat0);
    delete_mat(test_mat1);
    delete_mat(result_mat);
}

void test_2x3mul3x3()
{
    printf("testing 2x3*3x3 matmul\n");
    Mat *expected_mat = new_mat(2, 3);
    free(expected_mat->values);
    float expected[] = {15., 18., 21.,
                        42., 54., 66.};
    expected_mat->values = (float *)&expected;

    Mat *test_mat1 = new_mat(2, 3);
    free(test_mat1->values);
    float test_mat1_values[] = {0., 1., 2., 3., 4., 5.};

    Mat *test_mat0 = new_mat(3, 3);
    for (int i = 0; i < 9; i++)
        test_mat0->values[i] = i;
    test_mat1->values = (float *)&test_mat1_values;

    Mat *result_mat = new_mat(2, 3);
    mat_dot_mat(test_mat1, test_mat0, result_mat);
    // print_mat(result_mat);
    // print_mat(expected_mat);
    AssertTrue(equals(result_mat, expected_mat));

    delete_mat(expected_mat);
    delete_mat(test_mat0);
    delete_mat(test_mat1);
    delete_mat(result_mat);
}

void test_3x3_transform()
{
    printf("testing 3x3 transform\n");
    Mat *expected_mat = new_mat(3, 3);
    free(expected_mat->values);
    float expected[] = {0., 3., 6.,
                        1., 4., 7.,
                        2., 5., 8.};
    expected_mat->values = (float *)&expected;

    Mat *test_mat0 = new_mat(3, 3);
    for (int i = 0; i < 9; i++)
        test_mat0->values[i] = i;

    mat_T(test_mat0);
    // print_mat(expected_mat);
    AssertTrue(equals(test_mat0, expected_mat));

    delete_mat(expected_mat);
    delete_mat(test_mat0);
}

void test_3x2_transform()
{
    printf("testing 3x3 transform\n");
    Mat *expected_mat = new_mat(3, 2);
    free(expected_mat->values);
    float expected[] = {0., 3.,
                        1., 4.,
                        2., 5.};
    expected_mat->values = (float *)&expected;

    Mat *test_mat0 = new_mat(2, 3);
    for (int i = 0; i < 6; i++)
        test_mat0->values[i] = i;

    mat_T(test_mat0);
    // print_mat(expected_mat);
    // print_mat(test_mat0);
    AssertTrue(equals(test_mat0, expected_mat));

    delete_mat(expected_mat);
    delete_mat(test_mat0);
}

void test_saving_mat()
{
    printf("testing saving mat\n");
    Mat *mat = new_mat(100, 100);
    for (int i = 0; i < 10000; i++)
    {
        mat->values[i] = i;
    }
    // printf("\nJFKLD:SJKLFDS:\n");
    char name[] = "test";
    AssertTrue(save_mat(mat, (char *)&name) == 0);
    char path[] = "./models/model1757878495/test";
    printf("testing loading mat\n");
    Mat *result_mat = load_mat((char *)&path);
    AssertTrue(equals(mat, result_mat));
    delete_mat(result_mat);
    delete_mat(mat);
}


void test_mat_add_mat() {
    printf("testing mat_add_mat\n");
    Mat *a = new_mat(2, 2);
    Mat *b = new_mat(2, 2);
    for (int i = 0; i < 4; i++) {
        a->values[i] = i;
        b->values[i] = 10 + i;
    }
    Mat *out = new_mat(2, 2);
    mat_add_mat(a, b, out);
    float expected[] = {10, 12, 14, 16};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(b);
    delete_mat(out);
    delete_mat(expected_mat);
}

void test_mat_sub_mat() {
    printf("testing mat_sub_mat\n");
    Mat *a = new_mat(2, 2);
    Mat *b = new_mat(2, 2);
    for (int i = 0; i < 4; i++) {
        a->values[i] = 10 + i;
        b->values[i] = i;
    }
    Mat *out = new_mat(2, 2);
    mat_sub_mat(a, b, out);
    float expected[] = {10., 10., 10., 10.};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(b);
    delete_mat(out);
    delete_mat(expected_mat);
}

void test_mat_add_scalar() {
    printf("testing mat_add_scalar\n");
    Mat *a = new_mat(2, 2);
    for (int i = 0; i < 4; i++) a->values[i] = i;
    Mat *out = new_mat(2, 2);
    mat_add_scalar(a, 5.0f, out);
    float expected[] = {5, 6, 7, 8};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(out);
    delete_mat(expected_mat);
}

void test_mat_sub_scalar() {
    printf("testing mat_sub_scalar\n");
    Mat *a = new_mat(2, 2);
    for (int i = 0; i < 4; i++) a->values[i] = 10 + i;
    Mat *out = new_mat(2, 2);
    mat_sub_scalar(a, 5.0f, out);
    float expected[] = {5, 6, 7, 8};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(out);
    delete_mat(expected_mat);
}

void test_mat_mul_scalar() {
    printf("testing mat_mul_scalar\n");
    Mat *a = new_mat(2, 2);
    for (int i = 0; i < 4; i++) a->values[i] = i;
    Mat *out = new_mat(2, 2);
    mat_mul_scalar(a, 2.0f, out);
    float expected[] = {0, 2, 4, 6};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(out);
    delete_mat(expected_mat);
}

void test_mat_div_scalar() {
    printf("testing mat_div_scalar\n");
    Mat *a = new_mat(2, 2);
    for (int i = 0; i < 4; i++) a->values[i] = (i+1)*2;
    Mat *out = new_mat(2, 2);
    mat_div_scalar(a, 2.0f, out);
    float expected[] = {1, 2, 3, 4};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(out);
    delete_mat(expected_mat);
}

float square(float x) { return x * x; }

void test_mat_apply_fn() {
    printf("testing mat_apply_fn (square)\n");
    Mat *a = new_mat(2, 2);
    for (int i = 0; i < 4; i++) a->values[i] = i;
    Mat *out = new_mat(2, 2);
    mat_apply_fn(a, square, out);
    float expected[] = {0, 1, 4, 9};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(out);
    delete_mat(expected_mat);
}

void test_mat_hamard_mat() {
    printf("testing mat_hamard_mat\n");
    Mat *a = new_mat(2, 2);
    Mat *b = new_mat(2, 2);
    for (int i = 0; i < 4; i++) {
        a->values[i] = i+1;
        b->values[i] = 2*(i+1);
    }
    Mat *out = new_mat(2, 2);
    mat_hamard_mat(a, b, out);
    float expected[] = {2, 8, 18, 32};
    Mat *expected_mat = new_mat(2, 2);
    for (int i = 0; i < 4; i++) expected_mat->values[i] = expected[i];
    AssertTrue(equals(out, expected_mat));
    delete_mat(a);
    delete_mat(b);
    delete_mat(out);
    delete_mat(expected_mat);
}

void test_copy_mat() {
    printf("testing copy_mat\n");
    Mat *a = new_mat(2, 2);
    for (int i = 0; i < 4; i++) a->values[i] = i;
    Mat *b = copy_mat(a);
    AssertTrue(equals(a, b));
    delete_mat(a);
    delete_mat(b);
}

int main(int argc, char **argv)
{
    test_3x3mul3x3();
    test_3x3mul3x2();
    test_2x3mul3x3();
    test_3x3_transform();
    test_3x2_transform();
    // test_saving_mat();
    test_mat_add_mat();
    test_mat_sub_mat();
    test_mat_add_scalar();
    test_mat_sub_scalar();
    test_mat_mul_scalar();
    test_mat_div_scalar();
    test_mat_apply_fn();
    test_mat_hamard_mat();
    test_copy_mat();

    int Result = (GlobalFailedTests != 0);

    printf("Unit Tests %s: %d/%d passed.\n",
           Result ? "Failed" : "Successful",
           GlobalTotalTests - GlobalFailedTests,
           GlobalTotalTests);
}
