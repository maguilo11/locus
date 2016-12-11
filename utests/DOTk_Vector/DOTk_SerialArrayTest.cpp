/*
 * DOTk_SerialArrayTest.cpp
 *
 *  Created on: Apr 4, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_SerialArray.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkSerialArrayTest
{

TEST(DOTk_SerialArrayTest, size)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim);

    size_t result = array.size();
    EXPECT_EQ(1e4, result);
}

TEST(DOTk_SerialArrayTest, max)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, 1);

    array[0] = 2;
    double max = array.max();

    double tolerance = 1e-8;
    EXPECT_NEAR(2., max, tolerance);
}

TEST(DOTk_SerialArrayTest, min)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, 1.);

    array[0] = -2;
    double min = array.min();

    double tolerance = 1e-8;
    EXPECT_NEAR(-2., min, tolerance);
}

TEST(DOTk_SerialArrayTest, abs)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, -111.);

    array.abs();

    std::tr1::shared_ptr<dotk::Vector<double> > gold = array.clone();
    gold->fill(111.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, array, thread_count);
}

TEST(DOTk_SerialArrayTest, scale)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, 1.);

    array.scale(3.);

    std::tr1::shared_ptr<dotk::Vector<double> > gold = array.clone();
    gold->fill(3.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, array, thread_count);
}

TEST(DOTk_SerialArrayTest, cwiseProd)
{
    int dim = 1e4;
    dotk::StdArray<double> x(dim, 2.);
    dotk::StdArray<double> y(dim, 2.);

    x.cwiseProd(y);

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_SerialArrayTest, axpy)
{
    int dim = 1e4;
    dotk::StdArray<double> x(dim, 1.);
    dotk::StdArray<double> y(dim, 1.);

    y.axpy(3., x);

    std::tr1::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_SerialArrayTest, sum)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, 3.);

    double sum = array.sum();

    double tolerance = 1e-8;
    EXPECT_NEAR(3e4, sum, tolerance);
}

TEST(DOTk_SerialArrayTest, dot)
{
    int dim = 1e4;
    dotk::StdArray<double> x(dim, 1.);
    dotk::StdArray<double> y(dim, 1.);

    double dot = y.dot(x);

    double tolerance = 1e-8;
    EXPECT_NEAR(1e4, dot, tolerance);
}

TEST(DOTk_SerialArrayTest, norm)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, 1.);

    double norm = array.norm();

    double tolerance = 1e-8;
    EXPECT_NEAR(1e2, norm, tolerance);
}

TEST(DOTk_SerialArrayTest, copy)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, 1.);
    std::tr1::shared_ptr<dotk::Vector<double> > y = array.clone();

    y->copy(array);

    std::tr1::shared_ptr<dotk::Vector<double> > gold = array.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

TEST(DOTk_SerialArrayTest, gather)
{
    int dim = 1e4;
    dotk::StdArray<double> array(dim, 1.);
    std::vector<double> y(dim, 0);

    array.gather(y.data());

    std::tr1::shared_ptr<dotk::Vector<double> > gold = array.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(y.size(), y.data(), *gold, thread_count);
}

}
