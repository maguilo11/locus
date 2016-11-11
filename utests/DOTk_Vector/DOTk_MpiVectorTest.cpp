/*
 * DOTk_MpiVectorTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_MpiVector.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkMpiVectorTest
{

TEST(DOTk_MpiVectorTest, size)
{
    int dim = 1e4;
    dotk::MpiVector<double> vector(dim);

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

TEST(DOTk_MpiVectorTest, max)
{
    int dim = 1e4;
    dotk::MpiVector<double> vector(dim);
    vector.fill(1.);

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

TEST(DOTk_MpiVectorTest, min)
{
    int dim = 1e4;
    dotk::MpiVector<double> vector(MPI_COMM_WORLD, dim);
    vector.fill(1.);

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

TEST(DOTk_MpiVectorTest, abs)
{
    int dim = 1e4;
    double value = -1.;
    dotk::MpiVector<double> x(MPI_COMM_WORLD, dim, value);

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

    std::tr1::shared_ptr<dotk::vector<double> > gold = x.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiVectorTest, scale)
{
    int dim = 1e4;
    dotk::MpiVector<double> vector(MPI_COMM_WORLD, dim);
    vector.fill(1.);

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

    std::tr1::shared_ptr<dotk::vector<double> > gold = vector.clone();
    gold->fill(3.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_MpiVectorTest, cwiseProd)
{
    int dim = 1e4;
    dotk::MpiVector<double> x(MPI_COMM_WORLD, dim, 2.);
    dotk::MpiVector<double> y(MPI_COMM_WORLD, dim, 2.);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    x.cwiseProd(y);

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

    std::tr1::shared_ptr<dotk::vector<double> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiVectorTest, axpy)
{
    int dim = 1e4;
    dotk::MpiVector<double> x(MPI_COMM_WORLD, dim);
    dotk::MpiVector<double> y(MPI_COMM_WORLD, dim);

    x.fill(1.);
    y.fill(1.);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    y.axpy(3., x);

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

    std::tr1::shared_ptr<dotk::vector<double> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_MpiVectorTest, sum)
{
    int dim = 1e4;
    dotk::MpiVector<double> vector(MPI_COMM_WORLD, dim);

    vector.fill(3.);

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

TEST(DOTk_MpiVectorTest, dot)
{
    int dim = 1e4;
    dotk::MpiVector<double> x(MPI_COMM_WORLD, dim);
    dotk::MpiVector<double> y(MPI_COMM_WORLD, dim);

    x.fill(1.);
    y.fill(1.);

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

TEST(DOTk_MpiVectorTest, norm)
{
    int dim = 1e4;
    dotk::MpiVector<double> x(MPI_COMM_WORLD, dim);
    dotk::MpiVector<double> y(MPI_COMM_WORLD, dim);

    x.fill(1.);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    double norm = x.norm();

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

TEST(DOTk_MpiVectorTest, copy)
{
    int dim = 1e4;
    dotk::MpiVector<double> x(MPI_COMM_WORLD, dim);
    std::tr1::shared_ptr<dotk::vector<double> > y = x.clone();

    x.fill(1.);

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    y->copy(x);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
    }

    std::tr1::shared_ptr<dotk::vector<double> > gold = x.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

TEST(DOTk_MpiVectorTest, gather)
{
    int dim = 1e4;
    double value = 1.;
    dotk::MpiVector<double> x(MPI_COMM_WORLD, dim, value);
    std::vector<double> y(dim, 0.);
    EXPECT_EQ(dim, y.size());

    double start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    x.gather(y.data());

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);

        dotk::StdArray<double> gold(dim, 1.);
        int thread_count = 4;
        dotk::gtest::checkResults(y.size(), y.data(), gold, thread_count);
    }
}

}