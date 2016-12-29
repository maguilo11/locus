/*
 * DOTk_MpiX_ArrayTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_MpiX_Array.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkMpiXArrayTest
{

TEST(DOTk_MpiX_ArrayTest, size)
{
    int dim = 1e4;
    int thread_count = 4;
    dotk::MpiX_Array<double> vector(dim, thread_count);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    size_t result = vector.size();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    printf("The size of rank %d is %d\n", my_rank, dim);
    EXPECT_EQ(1e4, result);
}

TEST(DOTk_MpiX_ArrayTest, max)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        vector[0] = 2;
        start = MPI_Wtime();
    }

    double max = vector.max();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("The max value is %f\n", max);
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
        EXPECT_NEAR(2., max, tolerance);
    }
}

TEST(DOTk_MpiX_ArrayTest, min)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        vector[0] = -2;
        start = MPI_Wtime();
    }

    double min = vector.min();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("The min value is %f\n", min);
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
        EXPECT_NEAR(-2., min, tolerance);
    }
}

TEST(DOTk_MpiX_ArrayTest, abs)
{
    int dim = 1e4;
    double value = -1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> x(dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    x.abs();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("ABS WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiX_ArrayTest, scale)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    vector.scale(3.);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = vector.clone();
    gold->fill(3.);
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_MpiX_ArrayTest, elementWiseMultiplication)
{
    int dim = 1e4;
    int thread_count = 4;
    dotk::MpiX_Array<double> x(dim, thread_count, 2.);
    dotk::MpiX_Array<double> y(dim, thread_count, 2.);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    x.elementWiseMultiplication(y);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(4.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiX_ArrayTest, axpy)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> x(dim, thread_count, value);
    dotk::MpiX_Array<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    y.update(3., x, 1.);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(4.);
    dotk::gtest::checkResults(*gold, y);
}

TEST(DOTk_MpiX_ArrayTest, sum)
{
    int dim = 1e4;
    double value = 3.;
    int thread_count = 4;
    dotk::MpiX_Array<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    double sum = vector.sum();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("GSUM, value is %f\n", sum);
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
        EXPECT_NEAR(3e4, sum, tolerance);
    }
}

TEST(DOTk_MpiX_ArrayTest, dot)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> x(dim, thread_count, value);
    dotk::MpiX_Array<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    double dot = y.dot(x);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("GDOT, value is %f\n", dot);
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
        EXPECT_NEAR(1e4, dot, tolerance);
    }
}

TEST(DOTk_MpiX_ArrayTest, norm)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> vector(MPI_COMM_WORLD, dim, thread_count, value);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    double norm = vector.norm();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("GNORM, value is %f\n", norm);
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
        EXPECT_NEAR(1e2, norm, tolerance);
    }
}

TEST(DOTk_MpiX_ArrayTest, copy)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::MpiX_Array<double> x(MPI_COMM_WORLD, dim, thread_count, value);
    std::tr1::shared_ptr<dotk::Vector<double> > y = x.clone();

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    y->update(1., x, 0.);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

}
