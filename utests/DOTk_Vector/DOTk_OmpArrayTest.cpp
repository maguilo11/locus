/*
 * DOTk_OmpArrayTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_OmpArray.hpp"
#include "DOTk_OmpArray.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkOmpArrayTest
{

TEST(DOTk_OmpArrayTest, size)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    size_t result = vector.size();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        EXPECT_EQ(dotk::types::OMP_ARRAY, vector.type());
        printf("WTime is %f\n", time);
    }

    printf("The size of rank %d is %d\n", my_rank, dim);
    EXPECT_EQ(1e4, result);
}

TEST(DOTk_OmpArrayTest, max)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        vector[0] = 2;
        start = omp_get_wtime();
    }

    Real result = vector.max();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("The max value is %f\n", result);
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(2., result, tolerance);
    }
}

TEST(DOTk_OmpArrayTest, min)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        vector[0] = -2;
        start = omp_get_wtime();
    }

    Real result = vector.min();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("The min value is %f\n", result);
        printf("WTime is %f\n", time);
        Real tolerance = 1e-8;
        EXPECT_NEAR(-2., result, tolerance);
    }
}

TEST(DOTk_OmpArrayTest, abs)
{
    int dim = 1e4;
    Real value = -11.;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.abs();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("ABS WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(11.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_OmpArrayTest, scale)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> vector(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    vector.scale(3.);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpVectorTest, cwiseProd)
{
    int dim = 1e4;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, 2.);
    dotk::omp::array<Real> y(dim, thread_count, 2.);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.cwiseProd(y);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpArrayTest, axpy)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, value);
    dotk::omp::array<Real> y(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    y.axpy(3., x);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(4.);
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_OmpArrayTest, sum)
{
    int dim = 1e4;
    Real value = 3.;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    Real sum = x.sum();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpArrayTest, dot)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, value);
    dotk::omp::array<Real> y(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    Real dot = x.dot(y);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpArrayTest, norm)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    Real norm = x.norm();

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpArrayTest, copy)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, 0.);
    dotk::omp::array<Real> y(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.copy(y);

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_OmpArrayTest, gather)
{
    int dim = 1e4;
    Real value = 1.;
    int thread_count = 4;
    dotk::omp::array<Real> x(dim, thread_count, 0.);
    dotk::omp::array<Real> y(dim, thread_count, value);

    Real start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    y.gather(&(x[0]));

    Real finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    Real time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::vector<Real> > gold = y.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(x, *gold, thread_count);
}

}
