#include <math.h>
#define e 2.7182

float logistic(float x) {
    return 1/(1-pow(e, -1.*x));
}
