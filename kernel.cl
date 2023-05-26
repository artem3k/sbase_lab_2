#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void matrixMul(__global double* A, __global double* B, __global double* C, int width)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    double value = 0.0;
    for (int i = 0; i < width; ++i) {
        value += A[row * width + i] * B[i * width + col];
    }

    C[row * width + col] = value;
}
