/*
 * DOTk_OmpVectorTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_OmpVector.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkOmpVectorTest
{

TEST(DOTk_OmpVectorTest, size)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        vector[0] = 2;
        start = omp_get_wtime();
    }

    double result = vector.size();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("The max value is %f\n", result);
        printf("WTime is %f\n", time);
    }
}

TEST(DOTk_OmpVectorTest, max)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        vector[0] = 2;
        start = omp_get_wtime();
    }

    double result = vector.max();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("The max value is %f\n", result);
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
        EXPECT_NEAR(2., result, tolerance);
    }
}

TEST(DOTk_OmpVectorTest, min)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        vector[0] = -2;
        start = omp_get_wtime();
    }

    double result = vector.min();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("The min value is %f\n", result);
        printf("WTime is %f\n", time);
        double tolerance = 1e-8;
        EXPECT_NEAR(-2., result, tolerance);
    }
}

TEST(DOTk_OmpVectorTest, abs)
{
    int dim = 1e4;
    double value = -12.;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.abs();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("ABS WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(12);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_OmpVectorTest, scale)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    vector.scale(3.);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpVectorTest, cwiseProd)
{
    int dim = 1e4;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, 2.);
    dotk::OmpVector<double> y(dim, thread_count, 2.);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.cwiseProd(y);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpVectorTest, axpy)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, value);
    dotk::OmpVector<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    y.axpy(3., x);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(4.);
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_OmpVectorTest, sum)
{
    int dim = 1e4;
    double value = 3.;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    double sum = x.sum();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpVectorTest, dot)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, value);
    dotk::OmpVector<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    double dot = x.dot(y);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpVectorTest, norm)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    double norm = x.norm();

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
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

TEST(DOTk_OmpVectorTest, copy)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, 0.);
    dotk::OmpVector<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.copy(y);

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_OmpVectorTest, gather)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpVector<double> x(dim, thread_count, 0.);
    dotk::OmpVector<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    y.gather(&(x[0]));

    double finish = 0.;
    if(my_rank == 0)
    {
        finish = omp_get_wtime();
    }

    double time = finish - start;
    if(my_rank == 0)
    {
        printf("WTime is %f\n", time);
    }

    std::tr1::shared_ptr<dotk::Vector<double> > gold = y.clone();
    gold->fill(1.);
    dotk::gtest::checkResults(x, *gold, thread_count);
}

}
