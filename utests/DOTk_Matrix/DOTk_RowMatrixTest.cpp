/*
 * DOTk_RowMatrixTest.cpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mpi.h>
#include "gtest/gtest.h"

#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkRowMatrixTest
{

TEST(RowMatrix, Copy)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    EXPECT_TRUE(30u == matrix.size());
    EXPECT_TRUE(ncols == matrix.ncols());
    EXPECT_TRUE(nrows == matrix.nrows());

    Real gold[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.copy(matrix.size(), gold);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COPY WTime is %f\n", time);
        std::vector<Real> data(matrix.size());
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
    }
}

TEST(RowMatrix, RowMajorCopy)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    dotk::StdArray<Real> gold(ncols);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < nrows; ++index)
    {
        gold.fill(index);
        matrix.copy(index, gold);
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
        for(size_t index = 0; index < nrows; ++index)
        {
            gold.fill(index);
            dotk::gtest::checkResults(gold, *matrix.basis(index));
        }
    }
}

TEST(RowMatrix, ColumnMajorCopy)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    dotk::StdArray<Real> gold(nrows);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < ncols; ++index)
    {
        Real value = static_cast<Real>(index) + 1.;
        gold.fill(value);
        matrix.copy(index, gold, false);
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
        Real gold[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       1, 2, 3, 4 ,5, 6, 7, 8, 9, 10,
                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
    }
}

TEST(RowMatrix, RowMajorDot)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    std::vector<Real> results(nrows, 0.);
    dotk::StdArray<Real> input(ncols, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < nrows; ++index)
    {
        results.data()[index] = matrix.dot(index, input);
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
        Real tolerance = 1e-8;
        Real gold[] = {55, 155, 255};
        dotk::gtest::checkResults(nrows, gold, results.size(), results.data());
    }
}

TEST(RowMatrix, ColumnMajorDot)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    std::vector<Real> results(ncols, 0.);
    dotk::StdArray<Real> input(nrows, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < ncols; ++index)
    {
        results.data()[index] = matrix.dot(index, input, false);
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
        Real tolerance = 1e-8;
        Real gold[] = {33, 36, 39, 42, 45, 48, 51, 54, 57, 60};
        dotk::gtest::checkResults(ncols, gold, results.size(), results.data());
    }
}

TEST(RowMatrix, RowMajorNorm)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    std::vector<Real> input(ncols, 1.);
    std::vector<Real> results(nrows, 0.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < nrows; ++index)
    {
        results.data()[index] = matrix.norm(index);
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
        Real tolerance = 1e-8;
        Real gold[] = {19.621416870348, 49.849774322458, 81.148012914673};
        dotk::gtest::checkResults(nrows, gold, results.size(), results.data());
    }
}

TEST(RowMatrix, ColumnMajorNorm)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    std::vector<Real> results(ncols, 0.);
    std::vector<Real> input(nrows, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < ncols; ++index)
    {
        results.data()[index] = matrix.norm(index, false);
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
        Real tolerance = 1e-8;
        Real gold[] = {23.72762103540, 25.13961017995, 26.58947160061, 28.07133769523, 29.58039891549, 31.11269837220,
                       32.66496594212, 34.23448553724, 35.81898937714, 37.41657386773};
        dotk::gtest::checkResults(ncols, gold, results.size(), results.data());
    }
}

TEST(RowMatrix, RowMajorScale)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real alpha = 2;
    size_t row_index = 1;
    matrix.scale(row_index, alpha);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ROW MAJOR SCALE WTime is %f\n", time);
        Real gold[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       22, 24, 26, 28 ,30, 32, 34, 36, 38, 40,
                       21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, data.size(), data.data());
    }
}

TEST(RowMatrix, ColumnMajorScale)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real alpha = 2;
    size_t column_index = 3;
    matrix.scale(column_index, alpha, false);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COLUMN MAJOR SCALE WTime is %f\n", time);
        Real gold[] = {1, 2, 3, 8, 5, 6, 7, 8, 9, 10,
                       11, 12, 13, 28 ,15, 16, 17, 18, 19, 20,
                       21, 22, 23, 48, 25, 26, 27, 28, 29, 30};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, data.size(), data.data());
    }
}

TEST(RowMatrix, RowMajorAxpy)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input_data);

    Real alpha = 2.;
    dotk::StdArray<Real> input(ncols, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    for(size_t index = 0; index < nrows; ++index)
    {
        matrix.axpy(index, alpha, input);
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
        Real gold[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                       13, 14, 15, 16 ,17, 18, 19, 20, 21, 22,
                       23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, data.size(), data.data());
    }
}

TEST(RowMatrix, ColumnMajorAxpy)
{
    size_t nrows = 3;
    size_t ncols = 10;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input_data);

    Real alpha = 2.;
    size_t column_index = 8;
    dotk::StdArray<Real> input(nrows, 1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.axpy(column_index, alpha, input, false);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("COLUMN MAJOR AXPY WTime is %f\n", time);
        Real gold[] = {1, 2, 3, 4, 5, 6, 7, 8, 11, 10,
                       11, 12, 13, 14 ,15, 16, 17, 18, 21, 20,
                       21, 22, 23, 24, 25, 26, 27, 28, 31, 30};
        std::vector<Real> data(matrix.size());
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(matrix.size(), gold, data.size(), data.data());
    }
}

TEST(RowMatrix, Scale)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

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
        Real gold[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                       22, 24, 26, 28 ,30, 32, 34, 36, 38, 40,
                       42, 44, 46, 48, 50, 52, 54, 56, 58, 60};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, matrix.size(), data.data());
    }
}

TEST(RowMatrix, diag)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    dotk::StdArray<Real> diagonal(nrows);
    matrix.diag(diagonal);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("DIAG WTime is %f\n", time);
        Real gold[] = {1, 12, 23};
        dotk::gtest::checkResults(nrows, gold, diagonal);
    }
}

TEST(RowMatrix, setDiag)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    dotk::StdArray<Real> diagonal(nrows, 88.);
    matrix.setDiag(diagonal);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("SETDIAG WTime is %f\n", time);
        Real gold[] = {88, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 88, 13, 14 ,15, 16, 17, 18, 19, 20,
                       21, 22, 88, 24, 25, 26, 27, 28, 29, 30};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, data.size(), data.data());
    }
}

TEST(RowMatrix, scaleDiag)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.scaleDiag(2);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("SCALEDIAG WTime is %f\n", time);
        Real gold[] = {2, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       11, 24, 13, 14 ,15, 16, 17, 18, 19, 20,
                       21, 22, 46, 24, 25, 26, 27, 28, 29, 30};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, data.size(), data.data());
    }
}

TEST(RowMatrix, trace)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real value = matrix.trace();

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
        EXPECT_NEAR(36, value , tolerance);
    }
}

TEST(RowMatrix, Fill)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    matrix.fill(1);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("FILL WTime is %f\n", time);
        Real gold[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, data.size(), data.data());
    }
}

TEST(RowMatrix, Set)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real gold = 88;
    matrix.set(1,2,gold);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        Real value = 0.;
        printf("SetAndGet WTime is %f\n", value);
        Real tolerance = std::numeric_limits<Real>::epsilon();
        EXPECT_NEAR(gold, matrix(1,2), tolerance);
    }
}

TEST(RowMatrix, Basis)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);
    EXPECT_EQ(dotk::types::SERIAL_ROW_MATRIX, matrix.type());

    Real data[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                   22, 24, 26, 28 ,30, 32, 34, 36, 38, 40,
                   42, 44, 46, 48, 50, 52, 54, 56, 58, 60};
    matrix.copy(30, data);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        printf("COLUMN WTime is %f\n", 0.);
        Real gold_column_zero[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
        std::vector<Real> gold(gold_column_zero, gold_column_zero + ncols);
        dotk::gtest::checkResults(gold.size(), gold.data(), *matrix.basis(0));

        Real gold_column_one[] = {22, 24, 26, 28 ,30, 32, 34, 36, 38, 40};
        gold.assign(gold_column_one, gold_column_one + ncols);
        dotk::gtest::checkResults(gold.size(), gold.data(), *matrix.basis(1));

        Real gold_column_two[] = {42, 44, 46, 48, 50, 52, 54, 56, 58, 60};
        gold.assign(gold_column_two, gold_column_two + ncols);
        dotk::gtest::checkResults(gold.size(), gold.data(), *matrix.basis(2));
    }
}

TEST(RowMatrix, MatVecTranspose)
{
    size_t nrows = 3;
    size_t ncols = 10;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    dotk::StdArray<Real> input(nrows, 1.);
    dotk::StdArray<Real> output(ncols, 0.);

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
        printf("MATVEC TRANSPOSE WTime is %f\n", time);
        Real gold[] = {33, 36, 39, 42, 45, 48, 51, 54, 57, 60};
        dotk::gtest::checkResults(ncols, gold, output);
    }
}

TEST(RowMatrix, MatVec)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    dotk::StdArray<Real> input(ncols, 1.);
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
        Real gold[] = {55, 155, 255};
        dotk::gtest::checkResults(nrows, gold, output);
    }
}

TEST(RowMatrix, GemvTranspose)
{
    size_t ncols = 10;
    size_t nrows = 3;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    Real beta = 2;
    Real alpha = 2;
    dotk::StdArray<Real> output(ncols);
    dotk::StdArray<Real> input(nrows, 1.);
    for(size_t index = 0; index < ncols; ++index)
    {
        output[index] = index + 1;
    }

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
        printf("GEMV TRANSPOSE WTime is %f\n", time);
        Real gold[] = {68, 76, 84, 92, 100, 108, 116, 124, 132, 140};
        dotk::gtest::checkResults(ncols, gold, output);
    }
}

TEST(RowMatrix, Gemv)
{
    size_t nrows = 3;
    size_t ncols = 10;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, data);

    Real beta = 2;
    Real alpha = 2;
    dotk::StdArray<Real> output(nrows);
    dotk::StdArray<Real> input(ncols, 1.);
    output[0] = 1;
    output[1] = 2;
    output[2] = 3;

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
        Real gold[] = {112, 314, 516};
        dotk::gtest::checkResults(nrows, gold, output);
    }
}

TEST(DOTk_ColumnMatrix, identity)
{
    size_t nrows = 3;
    size_t ncols = 10;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> matrix(x, nrows);

    Real input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14 ,15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28, 29, 30};
    matrix.copy(30, input);

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
        Real gold[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 0, 0, 0, 0, 0, 0, 0};
        std::vector<Real> data(matrix.size(), 0.);
        matrix.gather(data.size(), data.data());
        dotk::gtest::checkResults(30, gold, data.size(), data.data());
    }
}

TEST(RowMatrix, Gemm1)
{
    size_t nrows = 2;
    size_t ncols = 2;
    dotk::StdArray<Real> x(ncols);
    dotk::serial::DOTk_RowMatrix<Real> A(x, nrows);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    dotk::serial::DOTk_RowMatrix<Real> B(x, nrows);
    B(0, 0) = 5;
    B(0, 1) = 6;
    B(1, 0) = 7;
    B(1, 1) = 8;
    dotk::serial::DOTk_RowMatrix<Real> C(x, nrows);

    // C = A*B
    A.gemm(false, false, 1., B, 0., C);
    std::vector<Real> gold(C.size());
    gold[0] = 19;
    gold[1] = 22;
    gold[2] = 43;
    gold[3] = 50;
    std::vector<Real> data(C.size(), 0.);
    C.gather(C.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold.data(), data.size(), data.data());

    // C = A^t*B
    C.fill(0.);
    A.gemm(true, false, 1., B, 0., C);
    gold.assign(gold.size(), 0.);
    gold[0] = 26;
    gold[1] = 30;
    gold[2] = 38;
    gold[3] = 44;
    data.assign(data.size(), 0.);
    C.gather(C.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold.data(), data.size(), data.data());

    // C = A*B^t
    C.fill(0.);
    A.gemm(false, true, 1., B, 0., C);
    gold.assign(gold.size(), 0.);
    gold[0] = 17;
    gold[1] = 23;
    gold[2] = 39;
    gold[3] = 53;
    data.assign(data.size(), 0.);
    C.gather(C.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold.data(), data.size(), data.data());

    // C = A^t*B^t
    C.fill(0.);
    A.gemm(true, true, 1., B, 0., C);
    gold.assign(gold.size(), 0.);
    gold[0] = 23;
    gold[1] = 31;
    gold[2] = 34;
    gold[3] = 46;
    data.assign(data.size(), 0.);
    C.gather(C.size(), data.data());
    dotk::gtest::checkResults(C.size(), gold.data(), data.size(), data.data());
}

TEST(RowMatrix, Gemm2)
{
    size_t A_nrows = 3;
    size_t A_ncols = 2;
    dotk::StdArray<Real> x(A_ncols);
    dotk::serial::DOTk_RowMatrix<Real> A(x, A_nrows);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    A(2, 0) = 5;
    A(2, 1) = 6;
    size_t B_nrows = 2;
    size_t B_ncols = 3;
    dotk::StdArray<Real> y(B_ncols);
    dotk::serial::DOTk_RowMatrix<Real> B(y, B_nrows);
    B(0, 0) = 1;
    B(0, 1) = 2;
    B(0, 2) = 3;
    B(1, 0) = 4;
    B(1, 1) = 5;
    B(1, 2) = 6;

    // C = A*B
    size_t C1_nrows = 3;
    size_t C1_ncols = 3;
    dotk::StdArray<Real> z1(C1_ncols);
    dotk::serial::DOTk_RowMatrix<Real> C1(z1, C1_nrows);
    A.gemm(false, false, 1., B, 0., C1);
    std::vector<Real> gold(C1.size());
    gold[0] = 9;
    gold[1] = 12;
    gold[2] = 15;
    gold[3] = 19;
    gold[4] = 26;
    gold[5] = 33;
    gold[6] = 29;
    gold[7] = 40;
    gold[8] = 51;
    std::vector<Real> data(C1.size(), 0.);
    C1.gather(C1.size(), data.data());
    dotk::gtest::checkResults(C1.size(), gold.data(), data.size(), data.data());

    // C = A*B'
    C1.fill(0.);
    A.gemm(false, true, 1., A, 0., C1);
    gold[0] = 5;
    gold[1] = 11;
    gold[2] = 17;
    gold[3] = 11;
    gold[4] = 25;
    gold[5] = 39;
    gold[6] = 17;
    gold[7] = 39;
    gold[8] = 61;
    data.assign(data.size(), 0.);
    C1.gather(C1.size(), data.data());
    dotk::gtest::checkResults(C1.size(), gold.data(), data.size(), data.data());

    // C = A'*B
    size_t C2_nrows = 2;
    size_t C2_ncols = 2;
    dotk::StdArray<Real> z2(C2_ncols);
    dotk::serial::DOTk_RowMatrix<Real> C2(z2, C2_nrows);
    A.gemm(true, false, 1., A, 0., C2);
    std::vector<Real> gold2(C2.size());
    gold2[0] = 35;
    gold2[1] = 44;
    gold2[2] = 44;
    gold2[3] = 56;
    std::vector<Real> data2(C2.size(), 0.);
    C2.gather(C2.size(), data2.data());
    dotk::gtest::checkResults(C2.size(), gold2.data(), data2.size(), data2.data());

    // C = A'*B'
    C2.fill(0.);
    A.gemm(true, true, 1., B, 0., C2);
    gold2[0] = 22;
    gold2[1] = 49;
    gold2[2] = 28;
    gold2[3] = 64;
    data2.assign(data2.size(), 0.);
    C2.gather(C2.size(), data2.data());
    dotk::gtest::checkResults(C2.size(), gold2.data(), data2.size(), data2.data());
}

TEST(RowMatrix, Gemm3)
{
    size_t A_nrows = 3;
    size_t A_ncols = 2;
    dotk::StdArray<Real> x(A_ncols);
    dotk::serial::DOTk_RowMatrix<Real> A(x, A_nrows);
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(1, 0) = 3;
    A(1, 1) = 4;
    A(2, 0) = 5;
    A(2, 1) = 6;
    size_t B_nrows = 2;
    size_t B_ncols = 3;
    dotk::StdArray<Real> y(B_ncols);
    dotk::serial::DOTk_RowMatrix<Real> B(y, B_nrows);
    B(0, 0) = 1;
    B(0, 1) = 2;
    B(0, 2) = 3;
    B(1, 0) = 4;
    B(1, 1) = 5;
    B(1, 2) = 6;

    // C = A*B
    size_t C1_nrows = 3;
    size_t C1_ncols = 3;
    dotk::StdArray<Real> z1(C1_ncols);
    dotk::serial::DOTk_RowMatrix<Real> C1(z1, C1_nrows);
    C1.fill(1.);
    A.gemm(false, false, 2., B, 1., C1);
    std::vector<Real> gold(C1.size());
    gold[0] = 19;
    gold[1] = 25;
    gold[2] = 31;
    gold[3] = 39;
    gold[4] = 53;
    gold[5] = 67;
    gold[6] = 59;
    gold[7] = 81;
    gold[8] = 103;
    std::vector<Real> data(C1.size(), 0.);
    C1.gather(C1.size(), data.data());
    dotk::gtest::checkResults(C1.size(), gold.data(), data.size(), data.data());

    // C = A*B'
    C1.fill(1.);
    A.gemm(false, true, 2., A, 1., C1);
    gold[0] = 11;
    gold[1] = 23;
    gold[2] = 35;
    gold[3] = 23;
    gold[4] = 51;
    gold[5] = 79;
    gold[6] = 35;
    gold[7] = 79;
    gold[8] = 123;
    data.assign(data.size(), 0.);
    C1.gather(C1.size(), data.data());
    dotk::gtest::checkResults(C1.size(), gold.data(), data.size(), data.data());

    // C = A'*B
    size_t C2_nrows = 2;
    size_t C2_ncols = 2;
    dotk::StdArray<Real> z2(C2_ncols);
    dotk::serial::DOTk_RowMatrix<Real> C2(z2, C2_nrows);
    C2.fill(1.);
    A.gemm(true, false, 2., A, 1., C2);
    std::vector<Real> gold2(C2.size());
    gold2[0] = 71;
    gold2[1] = 89;
    gold2[2] = 89;
    gold2[3] = 113;
    std::vector<Real> data2(C2.size(), 0.);
    C2.gather(C2.size(), data2.data());
    dotk::gtest::checkResults(C2.size(), gold2.data(), data2.size(), data2.data());

    // C = A'*B'
    C2.fill(1.);
    A.gemm(true, true, 2., B, 1., C2);
    gold2[0] = 45;
    gold2[1] = 99;
    gold2[2] = 57;
    gold2[3] = 129;
    data2.assign(data2.size(), 0.);
    C2.gather(C2.size(), data2.data());
    dotk::gtest::checkResults(C2.size(), gold2.data(), data2.size(), data2.data());
}

}
