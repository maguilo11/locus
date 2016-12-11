/*
 * DOTk_UpperTriangularMatrixTest.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mpi.h>
#include "gtest/gtest.h"

#include "DOTk_SerialArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_UpperTriangularMatrix.hpp"
#include "DOTk_UpperTriangularMatrix.cpp"

namespace DOTkUpperTriangularMatrixTest
{

TEST(UpperTriangularMatrix, Copy)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);
    EXPECT_EQ(21, matrix.size());
    EXPECT_EQ(nrows, matrix.nrows());
    EXPECT_EQ(nrows, matrix.ncols());
    EXPECT_EQ(dotk::types::SERIAL_UPPER_TRI_MATRIX, matrix.type());

    Real gold[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.copy(21, gold);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COPY WTime is %f\n", time);
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, RowMajorCopy)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    dotk::StdArray<Real> data(nrows, 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < nrows; ++index)
    {
        data.fill(index);
        matrix.copy(index, data);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ROW MAJOR COPY WTime is %f\n", time);
        Real gold[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, ColumnMajorCopy)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    dotk::StdArray<Real> column(nrows);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < nrows; ++index)
    {
        column.fill(index);
        matrix.copy(index, column, false);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COLUMN MAJOR COPY WTime is %f\n", time);
        Real gold[] = {0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, RowMajorNorm)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    std::vector<Real> results(nrows, 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t row = 0; row < nrows; ++row)
    {
        results[row] = matrix.norm(row);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ROW MAJOR NORM WTime is %f\n", time);
        Real gold[] = {9.539392014169, 20.371548787463, 27.092434368288, 29.478805945967, 27.586228448267, 21};
        dotk::gtest::checkResults(nrows, gold, results.size(), results.data());
    }
}

TEST(UpperTriangularMatrix, ColumnMajorNorm)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    std::vector<Real> results(nrows, 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    size_t ncols = matrix.ncols();
    for(size_t column = 0; column < ncols; ++column)
    {
        results[column] = matrix.norm(column, false);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COLUMN MAJOR NORM WTime is %f\n", time);
        Real gold[] = {1., 7.28010988928, 14.73091986265, 22.84731931759, 31.16087290176, 39.33192087859};
        dotk::gtest::checkResults(ncols, gold, results.size(), results.data());
    }
}

TEST(UpperTriangularMatrix, RowMajorScale)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t row = 0; row < nrows; ++row)
    {
        Real alpha = 1. + static_cast<Real>(row);
        matrix.scale(row, alpha);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ROW MAJOR SCALE WTime is %f\n", time);
        Real gold[] = {1, 2, 3, 4, 5, 6, 14, 16, 18, 20, 22, 36, 39, 42, 45, 64, 68, 72, 95, 100, 126};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, ColumnMajorScale)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t column = 0; column < nrows; ++column)
    {
        Real alpha = 1. + static_cast<Real>(column);
        matrix.scale(column, alpha, false);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COLUMN MAJOR SCALE WTime is %f\n", time);
        Real gold[] = {1, 4, 9, 16, 25, 36, 14, 24, 36, 50, 66, 36, 52, 70, 90, 64, 85, 108, 95, 120, 126};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, RowMajorAxpy)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    dotk::StdArray<Real> input(nrows, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t row = 0; row < nrows; ++row)
    {
        Real alpha = static_cast<Real>(row) + 1;
        matrix.axpy(row, alpha, input);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ROW MAJOR AXPY WTime is %f\n", time);
        Real gold[] = {2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 24, 25, 27};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, ColumnMajorAxpy)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    dotk::StdArray<Real> input(nrows, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t row = 0; row < nrows; ++row)
    {
        Real alpha = static_cast<Real>(row) + 1;
        matrix.axpy(row, alpha, input, false);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COLUMN MAJOR AXPY WTime is %f\n", time);
        Real gold[] = {2, 4, 6, 8, 10, 12, 9, 11, 13, 15, 17, 15, 17, 19, 21, 20, 22, 24, 24, 26, 27};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, RowMajorDot)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    dotk::StdArray<Real> results(nrows, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t row = 0; row < nrows; ++row)
    {
        results[row] = matrix.dot(row, results);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ROW MAJOR DOT WTime is %f\n", time);
        Real gold[] = {21, 45, 54, 51, 39, 21};
        dotk::gtest::checkResults(nrows, gold, results);
    }
}

TEST(UpperTriangularMatrix, ColumnMajorDot)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    std::vector<Real> results(nrows, 0.);
    dotk::StdArray<Real> input(nrows, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t column = 0; column < nrows; ++column)
    {
        results[column] = matrix.dot(column, input, false);
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COLUMN MAJOR DOT WTime is %f\n", time);
        Real gold[] = {1, 9, 23, 42, 65, 91};
        dotk::gtest::checkResults(nrows, gold, results.size(), results.data());
    }
}

TEST(UpperTriangularMatrix, Norm)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real result = matrix.norm();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("NORM WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(57.54128952326, result, tolerance);
    }
}

TEST(UpperTriangularMatrix, MatVec)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    dotk::StdArray<Real> input(nrows, 1.);
    dotk::StdArray<Real> output(nrows, 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.matVec(input, output);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("MATVEC WTime is %f\n", time);
        Real gold[] = {21, 45, 54, 51, 39, 21};
        dotk::gtest::checkResults(nrows, gold, output);
    }
}

TEST(UpperTriangularMatrix, MatVecTranspose)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    dotk::StdArray<Real> input(nrows, 1.);
    dotk::StdArray<Real> output(nrows, 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.matVec(input, output, true);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("MATVEC WTime is %f\n", time);
        Real gold[] = {1, 9, 23, 42, 65, 91};
        dotk::gtest::checkResults(nrows, gold, output);
    }
}

TEST(UpperTriangularMatrix, Gemv)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real beta = 2;
    Real alpha = 2;
    dotk::StdArray<Real> input(nrows, 1.);
    dotk::StdArray<Real> output(nrows, 0.);
    output[0] = 1.;
    output[1] = 2.;
    output[2] = 3.;
    output[3] = 4.;
    output[4] = 5.;
    output[5] = 6.;

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.gemv(alpha, input, beta, output);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("GEMV WTime is %f\n", time);
        Real gold[] = {44, 94, 114, 110, 88, 54};
        dotk::gtest::checkResults(nrows, gold, output);
    }
}

TEST(UpperTriangularMatrix, GemvTranspose)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real beta = 2;
    Real alpha = 2;
    dotk::StdArray<Real> input(nrows, 1.);
    dotk::StdArray<Real> output(nrows, 0.);
    output[0] = 1.;
    output[1] = 2.;
    output[2] = 3.;
    output[3] = 4.;
    output[4] = 5.;
    output[5] = 6.;

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.gemv(alpha, input, beta, output, true);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("GEMV WTime is %f\n", time);
        Real gold[] = {4, 22, 52, 92, 140, 194};
        dotk::gtest::checkResults(nrows, gold, output);
    }
}

TEST(UpperTriangularMatrix, Scale)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real alpha = 2.;
    matrix.scale(alpha);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("SCALE WTime is %f\n", time);
        Real gold[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, Fill)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real alpha = 2.;
    matrix.fill(alpha);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("FILL WTime is %f\n", time);
        Real gold[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, Gather)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);
    std::vector<Real> gather(matrix.size(), 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.gather(gather.size(), gather.data());

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("GATHER WTime is %f\n", time);
        Real gold[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
        dotk::gtest::checkResults(21, gold, gather.size(), gather.data());
    }
}

TEST(UpperTriangularMatrix, Diag)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);
    dotk::StdArray<Real> diag(nrows);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.diag(diag);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("DIAG WTime is %f\n", time);
        Real gold[] = {1, 7, 12, 16, 19, 21};
        dotk::gtest::checkResults(6, gold, diag);
    }
}

TEST(UpperTriangularMatrix, Shift)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.shift(2.);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("SHIFT WTime is %f\n", time);
        dotk::StdArray<Real> diag(nrows);
        matrix.diag(diag);
        Real gold[] = {3, 9, 14, 18, 21, 23};
        dotk::gtest::checkResults(6, gold, diag);

        Real matrix_gold[] = {3, 2, 3, 4, 5, 6, 9, 8, 9, 10, 11, 14, 13, 14, 15, 18, 17, 18, 21, 20, 23};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, matrix_gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, ScaleDiag)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.scaleDiag(2.);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("SCALEDIAG WTime is %f\n", time);
        Real matrix_gold[] = {2, 2, 3, 4, 5, 6, 14, 8, 9, 10, 11, 24, 13, 14, 15, 32, 17, 18, 38, 20, 42};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, matrix_gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, Trace)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    matrix.copy(21, data);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real result = matrix.trace();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("TRACE WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(76, result, tolerance);
    }
}

TEST(UpperTriangularMatrix, Set)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real value = 1;
    for(size_t row = 0; row < nrows; ++row)
    {
        for(size_t col = row; col < nrows; ++ col)
        {
            matrix.set(row,col,value);
            ++value;
        }
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("SET WTime is %f\n", time);
        Real gold[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, Get)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);

    std::vector<Real> gold(matrix.size(), 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real index = 0;
    for(size_t row = 0; row < nrows; ++row)
    {
        for(size_t col = row; col < nrows; ++ col)
        {
            gold[index] = matrix(row,col);
            ++index;
        }
    }

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("Get WTime is %f\n", time);
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(gold.size(), gold.data(), data.size(), data.data());
    }
}

TEST(UpperTriangularMatrix, Identity)
{
    size_t nrows = 6;
    dotk::serial::DOTk_UpperTriangularMatrix<Real> matrix(nrows);
    EXPECT_EQ(21, matrix.size());
    EXPECT_EQ(nrows, matrix.nrows());
    EXPECT_EQ(nrows, matrix.ncols());
    EXPECT_EQ(dotk::types::SERIAL_UPPER_TRI_MATRIX, matrix.type());

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.identity();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("IDENTITY WTime is %f\n", time);
        Real gold[] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(21, gold, data.size(), data.data());
    }
}

}
