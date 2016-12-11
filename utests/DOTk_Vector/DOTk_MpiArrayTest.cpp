/*
 * DOTk_MpiArrayTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_MpiArray.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkMpiArrayTest
{

TEST(DOTk_MpiArrayTest, size)
{
    int dim = 1e4;
    dotk::MpiArray<Real> array(dim);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    size_t result = array.size();

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

    printf("The size of rank %d is %d\n", my_rank, dim);
    EXPECT_EQ(1e4, result);
}

TEST(DOTk_MpiArrayTest, max)
{
    int dim = 1e4;
    Real value = 1.;
    dotk::MpiArray<Real> array(dim, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        array[0] = 2;
        start = MPI_Wtime();
    }

    Real max = array.max();

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

TEST(DOTk_MpiArrayTest, min)
{
    int dim = 1e4;
    Real value = 1.;
    dotk::MpiArray<Real> array(MPI_COMM_WORLD, dim, value);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        array[0] = -2;
        start = MPI_Wtime();
    }

    Real min = array.min();

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

TEST(DOTk_MpiArrayTest, abs)
{
    int dim = 1e4;
    Real value = -1.;
    dotk::MpiArray<Real> x(MPI_COMM_WORLD, dim, value);

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

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = x.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiArrayTest, scale)
{
    int dim = 1e4;
    dotk::MpiArray<Real> array(MPI_COMM_WORLD, dim);
    array.fill(1.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    array.scale(3.);

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

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = array.clone();
    gold->fill(3.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, array, thread_count);
}

TEST(DOTk_MpiArrayTest, cwiseProd)
{
    int dim = 1e4;
    dotk::MpiArray<Real> x(MPI_COMM_WORLD, dim, 2.);
    dotk::MpiArray<Real> y(MPI_COMM_WORLD, dim, 2.);

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

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_MpiArrayTest, axpy)
{
    int dim = 1e4;
    dotk::MpiArray<Real> x(MPI_COMM_WORLD, dim);
    dotk::MpiArray<Real> y(MPI_COMM_WORLD, dim);

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

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_MpiArrayTest, sum)
{
    int dim = 1e4;
    dotk::MpiArray<Real> array(MPI_COMM_WORLD, dim);

    array.fill(3.);

    Real start = 0.;
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if(my_rank == 0)
    {
        start = MPI_Wtime();
    }

    Real sum = array.sum();

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

TEST(DOTk_MpiArrayTest, dot)
{
    int dim = 1e4;
    dotk::MpiArray<Real> x(MPI_COMM_WORLD, dim);
    dotk::MpiArray<Real> y(MPI_COMM_WORLD, dim);

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

TEST(DOTk_MpiArrayTest, norm)
{
    int dim = 1e4;
    dotk::MpiArray<Real> x(MPI_COMM_WORLD, dim);
    dotk::MpiArray<Real> y(MPI_COMM_WORLD, dim);

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

TEST(DOTk_MpiArrayTest, copy)
{
    int dim = 1e4;
    Real value = 1.;
    dotk::MpiArray<Real> x(MPI_COMM_WORLD, dim, value);
    std::tr1::shared_ptr<dotk::Vector<Real> > y = x.clone();

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
    }

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = x.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

TEST(DOTk_MpiArrayTest, gather)
{
    size_t dim = 1e4;
    Real value = 1.;
    dotk::MpiArray<Real> x(MPI_COMM_WORLD, dim, value);
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

        dotk::StdArray<Real> gold(dim, 1.);
        int thread_count = 4;
        dotk::gtest::checkResults(y.size(), y.data(), gold, thread_count);
    }
}

}
