#include <math.h>
#include <stdlib.h>

double relu(double x) {
    return fmax(0.0, x);
}

void softmax(double* x, int length, double* result) {
    double max = x[0];
    for (int i = 1; i < length; ++i) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        result[i] = exp(x[i] - max);
        sum += result[i];
    }

    for (int i = 0; i < length; ++i) {
        result[i] /= sum;
    }
}