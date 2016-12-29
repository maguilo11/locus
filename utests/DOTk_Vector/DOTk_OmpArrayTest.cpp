/*
 * DOTk_OmpArrayTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_OmpArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkOmpArrayTest
{

TEST(DOTk_OmpArrayTest, size)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> vector(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    size_t result = vector.size();

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

    printf("The size of rank %d is %d\n", my_rank, dim);
    EXPECT_EQ(1e4, result);
}

TEST(DOTk_OmpArrayTest, max)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> vector(dim, thread_count, value);

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

TEST(DOTk_OmpArrayTest, min)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> vector(dim, thread_count, value);

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

TEST(DOTk_OmpArrayTest, abs)
{
    int dim = 1e4;
    double value = -11.;
    int thread_count = 4;
    dotk::OmpArray<double> x(dim, thread_count, value);

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
    gold->fill(11.);
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_OmpArrayTest, scale)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> vector(dim, thread_count, value);

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

TEST(DOTk_OmpVectorTest, elementWiseMultiplication)
{
    int dim = 1e4;
    int thread_count = 4;
    dotk::OmpArray<double> x(dim, thread_count, 2.);
    dotk::OmpArray<double> y(dim, thread_count, 2.);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.elementWiseMultiplication(y);

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

TEST(DOTk_OmpArrayTest, axpy)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> x(dim, thread_count, value);
    dotk::OmpArray<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    y.update(3., x, 1.);

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

TEST(DOTk_OmpArrayTest, sum)
{
    int dim = 1e4;
    double value = 3.;
    int thread_count = 4;
    dotk::OmpArray<double> x(dim, thread_count, value);

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

TEST(DOTk_OmpArrayTest, dot)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> x(dim, thread_count, value);
    dotk::OmpArray<double> y(dim, thread_count, value);

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

TEST(DOTk_OmpArrayTest, norm)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> x(dim, thread_count, value);

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

TEST(DOTk_OmpArrayTest, copy)
{
    int dim = 1e4;
    double value = 1.;
    int thread_count = 4;
    dotk::OmpArray<double> x(dim, thread_count, 0.);
    dotk::OmpArray<double> y(dim, thread_count, value);

    double start = 0.;
    int my_rank = omp_get_thread_num();
    if(my_rank == 0)
    {
        start = omp_get_wtime();
    }

    x.update(1., y, 0.);

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

}
