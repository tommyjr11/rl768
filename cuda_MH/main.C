#include <stdio.h>
#include "cuda_functions.h"

int main() {
    int size = 10;
    float data[10];
    
    for (int i = 0; i < size; i++) {
        data[i] = i + 1;
    }

    my_cuda_function(data, size);

    printf("Modified data:\n");
    for (int i = 0; i < size; i++) {
        printf("%f ", data[i]);
    }
    printf("\n");

    return 0;
}
