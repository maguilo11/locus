/*
 * DOTk_SerialVectorTest.cpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_SerialVector.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkSerialVectorTest
{

TEST(DOTk_SerialVectorTest, size)
{
    int dim = 1e4;
    dotk::StdVector<double> vector(dim);

    size_t result = vector.size();

    EXPECT_EQ(1e4, result);
}

TEST(DOTk_SerialVectorTest, max)
{
    int dim = 1e4;
    dotk::StdVector<double> vector(dim, 1);

    vector[0] = 2;
    double max = vector.max();

    double tolerance = 1e-8;
    EXPECT_NEAR(2., max, tolerance);
}

TEST(DOTk_SerialVectorTest, min)
{
    int dim = 1e4;
    dotk::StdVector<double> vector(dim, 1.);

    vector[0] = -2;
    double min = vector.min();

    double tolerance = 1e-8;
    EXPECT_NEAR(-2., min, tolerance);
}

TEST(DOTk_SerialVectorTest, abs)
{
    int dim = 1e4;
    std::vector<double> data(dim, -13.);
    dotk::StdVector<double> vector(data);

    vector.abs();

    std::shared_ptr<dotk::Vector<double> > gold = vector.clone();
    gold->fill(13.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_SerialVectorTest, scale)
{
    int dim = 1e4;
    std::vector<double> data(dim, 1.);
    dotk::StdVector<double> vector(data);

    vector.scale(3.);

    std::shared_ptr<dotk::Vector<double> > gold = vector.clone();
    gold->fill(3.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_SerialVectorTest, elementWiseMultiplication)
{
    int dim = 1e4;
    dotk::StdVector<double> x(dim, 2.);
    dotk::StdVector<double> y(dim, 2.);

    x.elementWiseMultiplication(y);

    std::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_SerialVectorTest, axpy)
{
    int dim = 1e4;
    dotk::StdVector<double> x(dim, 1.);
    dotk::StdVector<double> y(dim, 1.);

    y.update(3., x, 1.);

    std::shared_ptr<dotk::Vector<double> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_SerialVectorTest, sum)
{
    int dim = 1e4;
    dotk::StdVector<double> vector(dim, 3.);

    double sum = vector.sum();

    double tolerance = 1e-8;
    EXPECT_NEAR(3e4, sum, tolerance);
}

TEST(DOTk_SerialVectorTest, dot)
{
    int dim = 1e4;
    dotk::StdVector<double> x(dim, 1.);
    dotk::StdVector<double> y(dim, 1.);

    double dot = y.dot(x);

    double tolerance = 1e-8;
    EXPECT_NEAR(1e4, dot, tolerance);
}

TEST(DOTk_SerialVectorTest, norm)
{
    int dim = 1e4;
    dotk::StdVector<double> vector(dim, 1.);

    double norm = vector.norm();

    double tolerance = 1e-8;
    EXPECT_NEAR(1e2, norm, tolerance);
}

TEST(DOTk_SerialVectorTest, copy)
{
    int dim = 1e4;
    dotk::StdVector<double> vector(dim, 1.);
    std::shared_ptr<dotk::Vector<double> > y = vector.clone();

    y->update(1., vector, 0.);

    std::shared_ptr<dotk::Vector<double> > gold = vector.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

}
