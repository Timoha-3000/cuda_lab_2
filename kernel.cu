#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024 // Размер векторов
#define BLOCK_SIZE 256

// Ядро, приводящее к объединению запросов в одну транзакцию
static __global__ void multiplyCoalesced(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

// Ядро, не приводящее к объединению запросов в одну транзакцию
static __global__ void multiplyNonCoalesced(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int stride = gridDim.x * blockDim.x;
        for (int i = idx; i < n; i += stride) {
            C[i] = A[i] * B[i];
        }
    }
}

// Верификация результата
void verifyResult(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; ++i) {
        if (C[i] != A[i] * B[i]) {
            printf("Error at index %d: expected %d, got %d\n", i, A[i] * B[i], C[i]);
            return;
        }
    }
    printf("Verification passed!\n");
}

void multVetorsTask1() {
    int* h_A, * h_B, * h_C, * h_C_verif;
    int* d_A, * d_B, * d_C;
    size_t size = N * sizeof(int);

    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    h_C_verif = (int*)malloc(size);

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 100 + 1;
        h_B[i] = rand() % 100 + 1;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Настройка сетки и блоков
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Измерение времени выполнения с приводящим к объединению запросов в одну транзакцию ядром
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    multiplyCoalesced << <gridSize, blockSize >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float timeCoalesced;
    cudaEventElapsedTime(&timeCoalesced, start, stop);
    printf("Coalesced access time: %f ms\n", timeCoalesced);

    // Верификация
    verifyResult(h_A, h_B, h_C, N);

    // Измерение времени выполнения с не приводящим к объединению запросов в одну транзакцию ядром
    cudaEventRecord(start);
    multiplyNonCoalesced << <gridSize, blockSize >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float timeNonCoalesced;
    cudaEventElapsedTime(&timeNonCoalesced, start, stop);
    printf("Non-coalesced access time: %f ms\n", timeNonCoalesced);

    // Верификация
    verifyResult(h_A, h_B, h_C, N);

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_verif);
}

int main() {
    multVetorsTask1();

    return 0;
}