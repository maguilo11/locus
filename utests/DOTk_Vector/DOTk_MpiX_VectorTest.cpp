/*
 * DOTk_MpiX_VectorTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_MpiX_Vector.hpp"
#include "DOTk_MpiX_Vector.cpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_SerialArray.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkMpiXVectorTest
{

TEST(DOTk_MpiX_VectorTest, size)
{
    int dim = 1e4;
    int thread_count = 4;
    dotk::mpix::vector<Real> vector(dim, thread_count);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    size_t result = vector.size();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        EXPECT_EQ(dotk::types::MPIx_VECTOR, vector.type());
        printf("WTime is %f\n", time);
    }

    printf("The size of rank %d is %d\n", my_rank, dim);
    EXPECT_EQ(1e4, result);
}

TEST(DOTk_MpiX_VectorTest, max)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        vector[0] = 2;
        start = MPI_Wtime();
    }

    Real max = vector.max();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("The max value is %f\n", max);
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(2., max, tolerance);
    }
}

TEST(DOTk_MpiX_VectorTest, min)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        vector[0] = -2;
        start = MPI_Wtime();
    }

    Real min = vector.min();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("The min value is %f\n", min);
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(-2., min, tolerance);
    }
}

TEST(DOTk_MpiX_VectorTest, abs)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> x(dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    x.abs();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ABS WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiX_VectorTest, scale)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    vector.scale(3.);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = vector.clone();
    gold->fill(3.);
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_MpiX_VectorTest, cwiseProd)
{
    int dim = 1e4;
    int thread_count = 4;
    dotk::mpix::vector<Real> x(dim, thread_count, 2.);
    dotk::mpix::vector<Real> y(dim, thread_count, 2.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    x.cwiseProd(y);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(4.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiX_VectorTest, axpy)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> x(dim, thread_count, value);
    dotk::mpix::vector<Real> y(dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    y.axpy(3., x);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(4.);
    dotk::gtest::checkResults(*gold, y);
}

TEST(DOTk_MpiX_VectorTest, sum)
{
    int dim = 1e4;
    Real value = 3.;
    int thread_count = 4;
    dotk::mpix::vector<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real sum = vector.sum();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("GSUM, value is %f\n", sum);
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(3e4, sum, tolerance);
    }
}

TEST(DOTk_MpiX_VectorTest, dot)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> x(dim, thread_count, value);
    dotk::mpix::vector<Real> y(dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real dot = y.dot(x);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("GDOT, value is %f\n", dot);
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(1e4, dot, tolerance);
    }
}

TEST(DOTk_MpiX_VectorTest, norm)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> vector(MPI_COMM_WORLD, dim, thread_count, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real norm = vector.norm();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("GNORM, value is %f\n", norm);
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(1e2, norm, tolerance);
    }
}

TEST(DOTk_MpiX_VectorTest, copy)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> x(MPI_COMM_WORLD, dim, thread_count, value);
    std::tr1::shared_ptr<dotk::vector<Real> > y = x.clone();

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    y->copy(x);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

TEST(DOTk_MpiX_VectorTest, gather)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::mpix::vector<Real> x(MPI_COMM_WORLD, dim, thread_count, value);
    std::vector<Real> y(dim, 0.);
    EXPECT_EQ(dim, y.size());

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    x.gather(y.data());

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = MPI_Wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);

        dotk::serial::array<Real> gold(dim, 1.);
        int thread_count = 4;
        dotk::gtest::checkResults(y.size(), y.data(), gold, thread_count);
    }
}

}
