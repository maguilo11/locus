/*
 * DOTk_SerialDenseMatrixTest.cpp
 *
 *  Created on: Jul 17, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mpi.h>
#include "gtest/gtest.h"

#include "DOTk_SerialArray.hpp"
#include "DOTk_SerialArray.cpp"
#include "DOTk_DenseMatrix.hpp"
#include "DOTk_DenseMatrix.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkSerialDenseMatrixTest
{

TEST(SerialDenseMatrix, fill)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    EXPECT_EQ(3, matrix.ncols());
    EXPECT_EQ(3, matrix.nrows());
    EXPECT_EQ(9, matrix.size());
    EXPECT_EQ(dotk::types::SERIAL_DENSE_MATRIX, matrix.type());

    matrix.fill(3.);

    printf("FILL WTime is %f\n", 0.);
    Real gold[] = {3, 3, 3,
                   3, 3, 3,
                   3, 3, 3};
    dotk::serial::array<Real> data(matrix.size(), 3.);
    dotk::gtest::checkResults(matrix.size(), gold, data);
}

TEST(SerialDenseMatrix, scale)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows, 1.);

    matrix.scale(3.);

    printf("SCALE WTime is %f\n", 0.);
    Real gold[] = {3, 3, 3,
                   3, 3, 3,
                   3, 3, 3};
    dotk::serial::array<Real> data(matrix.size(), 3.);
    dotk::gtest::checkResults(matrix.size(), gold, data);
}

TEST(SerialDenseMatrix, identity)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);

    matrix.identity();

    printf("IDENTITY WTime is %f\n", 0.);
    Real gold[] = {1, 0, 0,
                   0, 1, 0,
                   0, 0, 1};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, diag)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    dotk::serial::array<Real> diagonal(nrows);
    matrix.diag(diagonal);

    printf("DIAG WTime is %f\n", 0.);
    Real gold[] = {1, 5, 9};
    dotk::gtest::checkResults(matrix.nrows(), gold, diagonal);
}

TEST(SerialDenseMatrix, shift)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    std::vector<Real> diagonal(nrows, 29.);
    matrix.shift(3);

    printf("SHIFT WTime is %f\n", 0.);
    Real gold[] = {4, 2, 3,
                   4, 8, 6,
                   7, 8, 12};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, setDiag)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    dotk::serial::array<Real> diagonal(nrows, 29.);
    matrix.setDiag(diagonal);

    printf("SETDIAG WTime is %f\n", 0.);
    Real gold[] = {29, 2, 3,
                   4, 29, 6,
                   7, 8, 29};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, scaleDiag)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    matrix.scaleDiag(2.);

    printf("SCALEDIAG WTime is %f\n", 0.);
    Real gold[] = {2, 2, 3,
                   4, 10, 6,
                   7, 8, 18};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, copy1)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    dotk::serial::DOTk_DenseMatrix<Real> copy(nrows, 8.);

    matrix.copy(copy);

    printf("COPY1 WTime is %f\n", 0.);
    Real gold[] = {8, 8, 8,
                   8, 8, 8,
                   8, 8, 8};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, copy2)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);

    std::vector<Real> copy(matrix.size(), 7.);
    matrix.copy(copy.size(), copy.data());

    printf("COPY2 WTime is %f\n", 0.);
    Real gold[] = {7, 7, 7,
                   7, 7, 7,
                   7, 7, 7};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, set)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);

    matrix.set(0, 0, 3.);

    printf("FILL WTime is %f\n", 0.);
    Real gold[] = {3, 0, 0,
                   0, 0, 0,
                   0, 0, 0};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, RowMajorCopy)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    dotk::serial::array<Real> input(nrows, 0.);
    for(size_t index = 0; index < nrows; ++index)
    {
        input.fill(index);
        matrix.copy(index, input);
    }

    printf("ROW MAJOR COPY WTime is %f\n", 0.);
    Real gold[] = {0, 0, 0,
                   1, 1, 1,
                   2, 2, 2};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, ColumnMajorCopy)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    dotk::serial::array<Real> input(nrows, 0.);
    for(size_t index = 0; index < nrows; ++index)
    {
        input.fill(index);
        matrix.copy(index, input, false);
    }

    printf("COLUMN MAJOR COPY WTime is %f\n", 0.);
    Real gold[] = {0, 1, 2,
                   0, 1, 2,
                   0, 1, 2};
    std::vector<Real> data(matrix.size(), 0.);
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, RowMajorNorm)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    std::vector<Real> results(nrows, 0.);
    for(size_t index = 0; index < nrows; ++index)
    {
        results[index] = matrix.norm(index);
    }

    printf("ROW MAJOR NORM WTime is %f\n", 0.);
    Real gold[] = {3.741657386773, 8.774964387392, 13.928388277184};
    dotk::gtest::checkResults(matrix.ncols(), gold, results.size(), results.data());
}

TEST(SerialDenseMatrix, ColumnMajorNorm)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    std::vector<Real> results(nrows, 0.);
    for(size_t index = 0; index < nrows; ++index)
    {
        results[index] = matrix.norm(index, false);
    }

    printf("COLUMN MAJOR NORM WTime is %f\n", 0.);
    Real gold[] = {8.124038404635, 9.643650760992, 11.224972160321};
    dotk::gtest::checkResults(matrix.ncols(), gold, results.size(), results.data());
}

TEST(SerialDenseMatrix, RowMajorScale)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    Real alpha = 1;
    for(size_t index = 0; index < nrows; ++index)
    {
        alpha += index;
        matrix.scale(index, alpha);
    }

    printf("ROW MAJOR SCALE WTime is %f\n", 0);
    std::vector<Real> data(matrix.size());
    Real gold[] = {1, 2, 3, 8, 10, 12, 28, 32, 36};
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, ColumnMajorScale)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    Real alpha = 1;
    for(size_t index = 0; index < nrows; ++index)
    {
        alpha += index;
        matrix.scale(index, alpha, false);
    }

    printf("COLUMN MAJOR SCALE WTime is %f\n", 0);
    std::vector<Real> data(matrix.size());
    Real gold[] = {1, 4, 12, 4, 10, 24, 7, 16, 36};
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, RowMajorAxpy)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    Real alpha = 2;
    dotk::serial::array<Real> input(nrows, 1.);
    for(size_t index = 0; index < nrows; ++index)
    {
        matrix.axpy(index, alpha, input);
    }

    printf("ROW MAJOR AXPY WTime is %f\n", 0);
    std::vector<Real> data(matrix.size());
    Real gold[] = {3, 4, 5, 6, 7, 8, 9, 10, 11};
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, ColumnMajorAxpy)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    Real alpha = 2;
    dotk::serial::array<Real> input(nrows, 1.);
    for(size_t index = 0; index < nrows; ++index)
    {
        matrix.axpy(index, alpha, input, false);
    }

    printf("COLUMN MAJOR AXPY WTime is %f\n", 0);
    std::vector<Real> data(matrix.size());
    Real gold[] = {3, 4, 5, 6, 7, 8, 9, 10, 11};
    matrix.gather(data.size(), data.data());
    dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
}

TEST(SerialDenseMatrix, RowMajorDot)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    std::vector<Real> results(nrows, 0.);
    dotk::serial::array<Real> input(nrows, 1.);
    for(size_t index = 0; index < results.size(); ++index)
    {
        results[index] = matrix.dot(index, input);
    }

    printf("ROW MAJOR DOT WTime is %f\n", 0.);
    std::vector<Real> gold(nrows, 0.);
    gold[0] = 6;
    gold[1] = 15;
    gold[2] = 24;
    dotk::gtest::checkResults(gold.size(), gold.data(), results.size(), results.data());
}

TEST(SerialDenseMatrix, ColumnMajorDot)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    std::vector<Real> results(nrows, 0.);
    dotk::serial::array<Real> input(nrows, 1.);
    for(size_t index = 0; index < results.size(); ++index)
    {
        results[index] = matrix.dot(index, input, false);
    }

    printf("COLUMN MAJOR DOT WTime is %f\n", 0.);
    std::vector<Real> gold(nrows, 0.);
    gold[0] = 12;
    gold[1] = 15;
    gold[2] = 18;
    dotk::gtest::checkResults(gold.size(), gold.data(), results.size(), results.data());
}

TEST(SerialDenseMatrix, trace)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    Real trace = matrix.trace();

    printf("TRACE WTime is %f\n", 0.);
    Real tolerance = 1e-8;
    EXPECT_NEAR(15., trace, tolerance);
}

TEST(SerialDenseMatrix, norm)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;

    Real norm = matrix.norm();

    printf("NORM WTime is %f\n", 0.);
    Real tolerance = 1e-8;
    EXPECT_NEAR(16.881943016134, norm, tolerance);
}

TEST(SerialDenseMatrix, MatVec)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;
    dotk::serial::array<Real> input(nrows, 1.);
    dotk::serial::array<Real> output(nrows, 0.);

    matrix.matVec(input, output);

    printf("MATVEC WTime is %f\n", 0.);
    std::vector<Real> gold(nrows);
    gold[0] = 6;
    gold[1] = 15;
    gold[2] = 24;
    dotk::gtest::checkResults(gold.size(), gold.data(), output);
}

TEST(SerialDenseMatrix, MatVecTranspose)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;
    dotk::serial::array<Real> input(nrows, 1.);
    dotk::serial::array<Real> output(nrows, 0.);

    matrix.matVec(input, output, true);

    printf("MATVEC TRANSPOSE WTime is %f\n", 0.);
    std::vector<Real> gold(nrows);
    gold[0] = 12;
    gold[1] = 15;
    gold[2] = 18;
    dotk::gtest::checkResults(gold.size(), gold.data(), output);
}

TEST(SerialDenseMatrix, Gemv)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;
    dotk::serial::array<Real> input(nrows, 1.);
    dotk::serial::array<Real> output(nrows, 1.);

    matrix.gemv(2., input, 1., output);

    printf("GEMV WTime is %f\n", 0.);
    std::vector<Real> gold(nrows);
    gold[0] = 13;
    gold[1] = 31;
    gold[2] = 49;
    dotk::gtest::checkResults(gold.size(), gold.data(), output);
}

TEST(SerialDenseMatrix, GemvTranspose)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> matrix(nrows);
    matrix(0,0) = 1; matrix(0,1) = 2; matrix(0,2) = 3;
    matrix(1,0) = 4; matrix(1,1) = 5; matrix(1,2) = 6;
    matrix(2,0) = 7; matrix(2,1) = 8; matrix(2,2) = 9;
    dotk::serial::array<Real> input(nrows, 1.);
    dotk::serial::array<Real> output(nrows, 1.);

    matrix.gemv(2., input, 1., output, true);

    printf("GEMV TRANSPOSE WTime is %f\n", 0.);
    std::vector<Real> gold(nrows);
    gold[0] = 25;
    gold[1] = 31;
    gold[2] = 37;
    dotk::gtest::checkResults(gold.size(), gold.data(), output);
}

TEST(SerialDenseMatrix, Gemm1)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> A(nrows);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;
    dotk::serial::DOTk_DenseMatrix<Real> B(nrows);
    B.copy(A);
    dotk::serial::DOTk_DenseMatrix<Real> C(nrows);

    printf("GEMM A*B WTime is %f\n", 0.);
    A.gemm(false, false, 1., B, 0., C);
    Real gold1[] = {30, 36, 42,
                    66, 81, 96,
                    102, 126, 150};
    std::vector<Real> data(C.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold1, data.size(), data.data());

    printf("GEMM A'*B WTime is %f\n", 0.);
    C.fill(0.);
    A.gemm(true, false, 1., B, 0., C);
    Real gold2[] = {66, 78, 90,
                    78, 93, 108,
                    90, 108, 126};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold2, data.size(), data.data());

    printf("GEMM A*B' WTime is %f\n", 0.);
    C.fill(0.);
    A.gemm(false, true, 1., B, 0., C);
    Real gold3[] = {14, 32, 50,
                    32, 77, 122,
                    50, 122, 194};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold3, data.size(), data.data());

    printf("GEMM A'*B' WTime is %f\n", 0.);
    C.fill(0.);
    A.gemm(true, true, 1., B, 0., C);
    Real gold4[] = {30, 66, 102,
                    36, 81, 126,
                    42, 96, 150};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold4, data.size(), data.data());
}

TEST(SerialDenseMatrix, Gemm2)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> A(nrows);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;
    dotk::serial::DOTk_DenseMatrix<Real> B(nrows);
    B.copy(A);
    dotk::serial::DOTk_DenseMatrix<Real> C(nrows);

    printf("GEMM A*B WTime is %f\n", 0.);
    A.gemm(false, false, 2., B, 0., C);
    Real gold1[] = {60, 72, 84,
                    132, 162, 192,
                    204, 252, 300};
    std::vector<Real> data(C.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold1, data.size(), data.data());

    printf("GEMM A'*B WTime is %f\n", 0.);
    C.fill(0.);
    A.gemm(true, false, 2., B, 0., C);
    Real gold2[] = {132, 156, 180,
                    156, 186, 216,
                    180, 216, 252};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold2, data.size(), data.data());

    printf("GEMM A*B' WTime is %f\n", 0.);
    C.fill(0.);
    A.gemm(false, true, 2., B, 0., C);
    Real gold3[] = {28, 64, 100,
                    64, 154, 244,
                    100, 244, 388};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold3, data.size(), data.data());

    printf("GEMM A'*B' WTime is %f\n", 0.);
    C.fill(0.);
    A.gemm(true, true, 2., B, 0., C);
    Real gold4[] = {60, 132, 204,
                    72, 162, 252,
                    84, 192, 300};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold4, data.size(), data.data());
}

TEST(SerialDenseMatrix, Gemm3)
{
    size_t nrows = 3;
    dotk::serial::DOTk_DenseMatrix<Real> A(nrows);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 4; A(1,1) = 5; A(1,2) = 6;
    A(2,0) = 7; A(2,1) = 8; A(2,2) = 9;
    dotk::serial::DOTk_DenseMatrix<Real> B(nrows);
    B.copy(A);
    dotk::serial::DOTk_DenseMatrix<Real> C(nrows, 1.);

    printf("GEMM A*B WTime is %f\n", 0.);
    A.gemm(false, false, 2., B, 1., C);
    Real gold1[] = {61, 73, 85,
                    133, 163, 193,
                    205, 253, 301};
    std::vector<Real> data(C.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold1, data.size(), data.data());

    printf("GEMM A'*B WTime is %f\n", 0.);
    C.fill(1.);
    A.gemm(true, false, 2., B, 1., C);
    Real gold2[] = {133, 157, 181,
                    157, 187, 217,
                    181, 217, 253};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold2, data.size(), data.data());

    printf("GEMM A*B' WTime is %f\n", 0.);
    C.fill(1.);
    A.gemm(false, true, 2., B, 1., C);
    Real gold3[] = {29, 65, 101,
                    65, 155, 245,
                    101, 245, 389};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold3, data.size(), data.data());

    printf("GEMM A'*B' WTime is %f\n", 0.);
    C.fill(1.);
    A.gemm(true, true, 2., B, 1., C);
    Real gold4[] = {61, 133, 205,
                    73, 163, 253,
                    85, 193, 301};
    data.assign(data.size(), 0.);
    C.gather(data.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold4, data.size(), data.data());
}

}
