#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <mkl.h>
#include <mkl_cblas.h>
#include <cmath>
#include <OpenCL/cl.h>
#define MAX_SOURCE_SIZE (0x100000)

//#include <OpenCL/opencl.h>
using namespace std;
const int SIZE = 5;


void printMatrixFirst10(double** result){
    cout << "-------------" << endl;
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            cout << result[i][j] << " ";
        }
        cout << endl;
    }
    cout << "-------------" << endl;
}

float MFlops(int milliseconds){
    float seconds = (float)milliseconds / 1000.0;
    float c = 2.0 * pow(SIZE,3);
    return c / seconds * pow(10,-6);
}

double **allocateMatrix()
{
    double **matrix = nullptr;
    matrix = (double **) mkl_malloc(SIZE* sizeof(double*), 64);
    for (int i = 0; i < SIZE; ++i)
    {
        matrix[i] = (double*) mkl_malloc(SIZE* sizeof(double), 64);
    }

    return matrix;
}
/*
 * double **a = new double*(SIZE);
 * double *mem = new double(SIZE*SIZE);
 * for (int i = 0; i<SIZE; i++){
 *      a[i] = mem;
 *      mem+=SIZE;
 * }
 * a[i][j]
 * delete [] a[0];
 * delete [] a;
 * mem(n*i+j)
 *
 */
void fillMatrix(double **matrix)
{
    srand(time(NULL));
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            matrix[i][j] = (double) rand() / RAND_MAX;
        }
    }
}

void freeMatrix(double **matrix)
{
    for (int i = 0; i < SIZE; ++i)
    {
        mkl_free(matrix[i]);
    }

    mkl_free(matrix);
}

double **multiplyMatrix(double **matrix1, double **matrix2)
{
    double **result = allocateMatrix();
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < SIZE; ++k)
            {
                sum += matrix1[i][k] *matrix2[k][j];
            }

            result[i][j] = sum;
        }
    }

    return result;
}

char* readKernelSource(const char* filename)
{
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Failed to open kernel file\n");
        exit(-1);
    }

    if (fseek(file, 0, SEEK_END)) {
        fprintf(stderr, "Failed to seek kernel file\n");
        exit(-1);
    }

    long size = ftell(file);
    if (size == -1) {
        fprintf(stderr, "Failed to get size of kernel file\n");
        exit(-1);
    }

    if (fseek(file, 0, SEEK_SET)) {
        fprintf(stderr, "Failed to seek kernel file\n");
        exit(-1);
    }

    char* source = (char*) malloc(size + 1);
    if (!source) {
        fprintf(stderr, "Failed to allocate memory for kernel source\n");
        exit(-1);
    }

    fread(source, 1, size, file);
    source[size] = '\0';

    fclose(file);

    return source;
}

double **multiplyMatrix3(double **matrix1, double **matrix2)
{
    // Преобразование двумерного массива в одномерный для использования с OpenCL
    double *flat_matrix1 = (double*) malloc(SIZE * SIZE * sizeof(double));
    double *flat_matrix2 = (double*) malloc(SIZE * SIZE * sizeof(double));

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            flat_matrix1[i * SIZE + j] = matrix1[i][j];
            flat_matrix2[i * SIZE + j] = matrix2[i][j];
        }
    }

    // Использование OpenCL для умножения матриц

    // Загрузка и компиляция ядра
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    char *kernelSource = readKernelSource("../kernel.cl"); // предполагается, что файл kernel.cl содержит ваш код OpenCL
    program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "matrixMul", &err);

    // Создание буферов
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * SIZE * sizeof(double), NULL, NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE * SIZE * sizeof(double), NULL, NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * SIZE * sizeof(double), NULL, NULL);

    // Копирование матриц в буферы
    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, SIZE * SIZE * sizeof(double), flat_matrix1, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, SIZE * SIZE * sizeof(double), flat_matrix2, 0, NULL, NULL);

    // Установка аргументов ядра
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &SIZE);

    // Запуск ядра
    size_t globalWorkSize[2] = {SIZE, SIZE};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    // Чтение результата
    double *flat_result = (double*) malloc(SIZE * SIZE * sizeof(double));
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, SIZE * SIZE * sizeof(double), flat_result, 0, NULL, NULL);

    // Очистка
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Возвращение результата в формате двумерного массива
    double **result = allocateMatrix();
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            result[i][j] = flat_result[i * SIZE + j];
        }
    }

    // Очистка памяти
    free(flat_matrix1);
    free(flat_matrix2);
    free(flat_result);

    return result;
}



void multiplyMatrixBLAS(double **matrix1, double **matrix2, double **result)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, SIZE, SIZE, SIZE, 1.0, *matrix1, SIZE, *matrix2, SIZE, 0.0, *result, SIZE);
}

int main()
{
    cout << "Лабораторная работа №2\n" << "Круглый Артём Васильевич ФИТУ АИСа-о22 \n";

    double **matrix1 = allocateMatrix();
    double **matrix2 = allocateMatrix();
    fillMatrix(matrix1);
    fillMatrix(matrix2);
    auto start1 = chrono::high_resolution_clock::now();
    double **result1 = multiplyMatrix(matrix1, matrix2);
    auto end1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds > (end1 - start1);
    int milliseconds1 = duration1.count();
    cout << "1) Time taken: " << milliseconds1 << " milliseconds; " << "MFlops: " << MFlops(milliseconds1) << endl;


    double *matrix1_flat = (double*) malloc(SIZE *SIZE* sizeof(double));
    double *matrix2_flat = (double*) malloc(SIZE *SIZE* sizeof(double));
    double *result2_flat = (double*) malloc(SIZE *SIZE* sizeof(double));
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            matrix1_flat[i *SIZE + j] = matrix1[i][j];
            matrix2_flat[i *SIZE + j] = matrix2[i][j];
        }
    }

    auto start2 = chrono::high_resolution_clock::now();
    multiplyMatrixBLAS(&matrix1_flat, &matrix2_flat, &result2_flat);
    auto end2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds > (end2 - start2);
    int milliseconds2 = duration2.count();
    cout << "2) Time taken: " << milliseconds2 << " milliseconds; " << "MFlops: " << MFlops(milliseconds2) << endl;;

    double **result2 = allocateMatrix();

    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            result2[i][j] = result2_flat[i *SIZE + j];
        }
    }

    auto start3 = chrono::high_resolution_clock::now();
    double **result3 = multiplyMatrix3(matrix1, matrix2);
    auto end3 = chrono::high_resolution_clock::now();
    auto duration3 = chrono::duration_cast<chrono::milliseconds > (end3 - start3);
    int milliseconds3 = duration3.count();
    cout << "3) Time taken: " << milliseconds3 << " milliseconds; " << "MFlops: " << MFlops(milliseconds3) << endl;

    printMatrixFirst10(result1);
    printMatrixFirst10(result2);
    printMatrixFirst10(result3);

    free(result2_flat);
    free(matrix2_flat);
    free(matrix1_flat);
    freeMatrix(result1);
    freeMatrix(result2);
    freeMatrix(matrix1);
    freeMatrix(matrix2);
    return 0;
}