/*
 * DOTk_MpiVectorTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_MpiVector.hpp"
#include "DOTk_MpiVector.cpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_SerialArray.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkMpiVectorTest
{

TEST(DOTk_MpiVectorTest, size)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> vector(dim);

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
        EXPECT_EQ(dotk::types::MPI_VECTOR, vector.type());
        printf("WTime is %f\n", time);
    }

    printf("The size of rank %d is %d\n", my_rank, dim);
    EXPECT_EQ(1e4, result);
}

TEST(DOTk_MpiVectorTest, max)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> vector(dim);
    vector.fill(1.);

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

TEST(DOTk_MpiVectorTest, min)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> vector(MPI_COMM_WORLD, dim);
    vector.fill(1.);

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

TEST(DOTk_MpiVectorTest, abs)
{
    int dim = 1e4;
    Real value = -1.;
    dotk::mpi::vector<Real> x(MPI_COMM_WORLD, dim, value);

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
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiVectorTest, scale)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> vector(MPI_COMM_WORLD, dim);
    vector.fill(1.);

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
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_MpiVectorTest, cwiseProd)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> x(MPI_COMM_WORLD, dim, 2.);
    dotk::mpi::vector<Real> y(MPI_COMM_WORLD, dim, 2.);

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
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiVectorTest, axpy)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> x(MPI_COMM_WORLD, dim);
    dotk::mpi::vector<Real> y(MPI_COMM_WORLD, dim);

    x.fill(1.);
    y.fill(1.);

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
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_MpiVectorTest, sum)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> vector(MPI_COMM_WORLD, dim);

    vector.fill(3.);

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

TEST(DOTk_MpiVectorTest, dot)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> x(MPI_COMM_WORLD, dim);
    dotk::mpi::vector<Real> y(MPI_COMM_WORLD, dim);

    x.fill(1.);
    y.fill(1.);

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

TEST(DOTk_MpiVectorTest, norm)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> x(MPI_COMM_WORLD, dim);
    dotk::mpi::vector<Real> y(MPI_COMM_WORLD, dim);

    x.fill(1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real norm = x.norm();

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

TEST(DOTk_MpiVectorTest, copy)
{
    int dim = 1e4;
    dotk::mpi::vector<Real> x(MPI_COMM_WORLD, dim);
    std::tr1::shared_ptr<dotk::vector<Real> > y = x.clone();

    x.fill(1.);

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
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

TEST(DOTk_MpiVectorTest, gather)
{
    int dim = 1e4;
    Real value = 1.;
    dotk::mpi::vector<Real> x(MPI_COMM_WORLD, dim, value);
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
