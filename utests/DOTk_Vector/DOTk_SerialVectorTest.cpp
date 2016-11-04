/*
 * DOTk_SerialVectorTest.cpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_SerialVector.hpp"
#include "DOTk_SerialVector.cpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkSerialVectorTest
{

TEST(DOTk_SerialVectorTest, size)
{
    int dim = 1e4;
    dotk::serial::vector<Real> vector(dim);

    size_t result = vector.size();

    EXPECT_EQ(dotk::types::SERIAL_VECTOR, vector.type());

    EXPECT_EQ(1e4, result);
}

TEST(DOTk_SerialVectorTest, max)
{
    int dim = 1e4;
    dotk::serial::vector<Real> vector(dim, 1);

    vector[0] = 2;
    Real max = vector.max();

    Real tolerance = 1e-8;
    EXPECT_NEAR(2., max, tolerance);
}

TEST(DOTk_SerialVectorTest, min)
{
    int dim = 1e4;
    dotk::serial::vector<Real> vector(dim, 1.);

    vector[0] = -2;
    Real min = vector.min();

    Real tolerance = 1e-8;
    EXPECT_NEAR(-2., min, tolerance);
}

TEST(DOTk_SerialVectorTest, abs)
{
    int dim = 1e4;
    std::vector<Real> data(dim, -13.);
    dotk::serial::vector<Real> vector(data);

    vector.abs();

    std::tr1::shared_ptr<dotk::vector<Real> > gold = vector.clone();
    gold->fill(13.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_SerialVectorTest, scale)
{
    int dim = 1e4;
    std::vector<Real> data(dim, 1.);
    dotk::serial::vector<Real> vector(data);

    vector.scale(3.);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = vector.clone();
    gold->fill(3.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, vector, thread_count);
}

TEST(DOTk_SerialVectorTest, cwiseProd)
{
    int dim = 1e4;
    dotk::serial::vector<Real> x(dim, 2.);
    dotk::serial::vector<Real> y(dim, 2.);

    x.cwiseProd(y);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, x, thread_count);
}

TEST(DOTk_SerialVectorTest, axpy)
{
    int dim = 1e4;
    dotk::serial::vector<Real> x(dim, 1.);
    dotk::serial::vector<Real> y(dim, 1.);

    y.axpy(3., x);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = x.clone();
    gold->fill(4.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, y, thread_count);
}

TEST(DOTk_SerialVectorTest, sum)
{
    int dim = 1e4;
    dotk::serial::vector<Real> vector(dim, 3.);

    Real sum = vector.sum();

    Real tolerance = 1e-8;
    EXPECT_NEAR(3e4, sum, tolerance);
}

TEST(DOTk_SerialVectorTest, dot)
{
    int dim = 1e4;
    dotk::serial::vector<Real> x(dim, 1.);
    dotk::serial::vector<Real> y(dim, 1.);

    Real dot = y.dot(x);

    Real tolerance = 1e-8;
    EXPECT_NEAR(1e4, dot, tolerance);
}

TEST(DOTk_SerialVectorTest, norm)
{
    int dim = 1e4;
    dotk::serial::vector<Real> vector(dim, 1.);

    Real norm = vector.norm();

    Real tolerance = 1e-8;
    EXPECT_NEAR(1e2, norm, tolerance);
}

TEST(DOTk_SerialVectorTest, copy)
{
    int dim = 1e4;
    dotk::serial::vector<Real> vector(dim, 1.);
    std::tr1::shared_ptr<dotk::vector<Real> > y = vector.clone();

    y->copy(vector);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = vector.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(*gold, *y, thread_count);
}

TEST(DOTk_SerialVectorTest, gather)
{
    int dim = 1e4;
    dotk::serial::vector<Real> vector(dim, 1.);
    std::vector<Real> y(dim, 0);

    vector.gather(y.data());

    std::tr1::shared_ptr<dotk::vector<Real> > gold = vector.clone();
    gold->fill(1.);
    int thread_count = 4;
    dotk::gtest::checkResults(y.size(), y.data(), *gold, thread_count);
}

}
