#include <iostream>
#include <complex>
#include <mkl.h>
#include <immintrin.h>

using std::complex;
using std::cout;
using std::cin;
using std::endl;


typedef complex<float> sComplex;

const int N = 2048;

sComplex* massiveMatrix(const int n)
{
  sComplex* matrixArrMassive = new sComplex[n * n];
  return matrixArrMassive;
}

void deleteMatrix(sComplex* &m)
{
  delete[] m;//Эта строка удаляет пул памяти, выделенный для элементов матрицы.
}


void matrixTransp(const int n, sComplex* a, sComplex* at)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        at[j * n + i] = a[i * j + 1];
}

void matrixMultiplication(const int n, sComplex* a, sComplex* b, sComplex* c)
{
#pragma omp parallel for
    sComplex s = 0;
    for (int i = 0; i < n; ++i) 
    {
        for (int j = 0; j < n; ++j)
        {
            s = 0;
            for (int k = 0; k < n; k++) {
                s += a[i * n + k] * b[j * n + k];
            }
            c[i * n + j] = s;
        }
    }
}

void matrixOptimizedMultiplication(sComplex* A, sComplex* B, sComplex* C, int blockSize)
{
    int numBlocks = N / blockSize;

#pragma omp parallel for collapse(2) num_threads(128)
    sComplex s = 0;
    for (int iBlock = 0; iBlock < numBlocks; ++iBlock)
    {
        for (int jBlock = 0; jBlock < numBlocks; ++jBlock)
        {
            for (int kBlock = 0; kBlock < numBlocks; ++kBlock)
            {
                // Умножение блоков размером blockSize x blockSize
                for (int i = 0; i < blockSize; ++i)
                {
                    int iOffset = iBlock * blockSize + i;
                    for (int j = 0; j < blockSize; ++j)
                    {
                        int jOffset = jBlock * blockSize + j;
                         s = 0;
                        for (int k = 0; k < blockSize; ++k)
                        {
                            s += A[iOffset * N + kBlock * blockSize + k] *
                                B[jOffset * N + kBlock * blockSize + k];
                        }
                        C[iOffset * N + jOffset] = s;
                    }
                }
            }
        }
    }
}

bool matrixChech(const int n, sComplex* a, sComplex* b)
{
    float eps = 1.e-3;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (std::abs(a[i * n + j] - b[i * n + j]) > eps)
                return false;
    return true;
}

int main()
{
    setlocale(LC_ALL, "russian");
    sComplex* A = massiveMatrix(N);
    sComplex* B = massiveMatrix(N);
    sComplex* Bt = massiveMatrix(N);
    sComplex* C1 = massiveMatrix(N);
    sComplex* C2 = massiveMatrix(N);
    sComplex* C3 = massiveMatrix(N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            A[i * N + j].real((float)rand() / RAND_MAX);
            A[i * N + j].imag(float(rand()) / RAND_MAX);
            B[i * N + j].real((float)rand() / RAND_MAX);
            B[i * N + j].imag(float(rand()) / RAND_MAX);
        }
    cout << "a[2][2] = " << A[2 * N + 2] << endl;
    cout << "b[2][2] = " << B[2 * N + 2] << endl;
    cout << "bt[2][2] = " << Bt[2 * N + 2] << endl;
    clock_t start, end;
    start = clock();

    matrixTransp(N, B, Bt);
    matrixMultiplication(N, A, Bt, C1);

    end = clock();

    double elapsed_secs;
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "линейное перемножение: " << elapsed_secs << " - время работы  алгоритма , p = " << 2.0 * (double)N * N * N / elapsed_secs * 1.e-6 << " MFlops\n";

    cout << "c1[2][2] = " << C1[2 * N + 2] << endl;
    //////////////////////////////////////////////////////////////////////////////////////

    sComplex alpha(1, 0);
    sComplex beta(0, 0);
    start = clock();
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha, &A[0 * N + 0], N, &Bt[0], N, &beta, &C2[0], N);
    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "перемножение с помощью библиотеки MKL: " << elapsed_secs << " - время работы  алгоритма, p = " << 2.0 * (double)N * (double)N * (double)N / elapsed_secs * 1.e-6 << " MFlops\n";

    cout << "a[2][2] = " << A[2 * N + 2] << endl;
    cout << "b[2][2] = " << B[2 * N + 2] << endl;
    cout << "bt[2][2] = " << Bt[2 * N + 2] << endl;
    cout << "c2[2][2] = " << C2[2 * N + 2] << endl;

    cout << "question: c1 = c2? answer: " << matrixChech(N, C1, C2) << endl;


    //////////////////////////////////////////////////////////////////////////////////////
    start = clock();

    //matrixTransp(N, B, Bt);
    matrixOptimizedMultiplication(A, Bt, C3, 64);

    end = clock();
    elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
    cout << "оптимизированное перемножение: " << elapsed_secs << " - время работы  алгоритма , p = " << 2.0 * (double)N * N * N / elapsed_secs * 1.e-6 << " MFlops\n";

    cout << "a[2][2] = " << A[2 * N + 2] << endl;
    cout << "b[2][2] = " << B[2 * N + 2] << endl;
    cout << "bt[2][2] = " << Bt[2 * N + 2] << endl;
    cout << "c3[2][2] = " << C3[2 * N + 2] << endl;

    cout << "question: c1 = c3? answer: " << matrixChech(N, C1, C3) << endl;//вот здесь сравнение выдает false, вероятно сам метод некорректный.


    /////////////////////////////////////////////////////////////////////////////////////
    deleteMatrix(A);
    deleteMatrix(B);
    deleteMatrix(Bt);
    deleteMatrix(C1);
    deleteMatrix(C2);
    deleteMatrix(C3);
    return 0;
}
