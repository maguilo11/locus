/*
 * DOTk_FreeFunctionsTest.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_MathUtils.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_ColumnMatrix.hpp"
#include "DOTk_ColumnMatrix.cpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GradBasedIoUtils.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_DescentDirectionTools.hpp"

namespace DOTkFreeFunctionsTest
{

TEST(DOTk_MATH, givens)
{
    Real cosine, sine;
    // TEST 1: a > b
    dotk::givens(3.003240019013723, 2.490569770109506, cosine, sine);
    const Real tol = 1e-10;
    EXPECT_NEAR(0.769748131097168, cosine, tol);
    EXPECT_NEAR(0.638347722383669, sine, tol);
    // TEST 2: b > a
    dotk::givens(2.490569770109506, 3.003240019013723, cosine, sine);
    EXPECT_NEAR(0.638347722383669, cosine, tol);
    EXPECT_NEAR(0.769748131097168, sine, tol);
    // TEST 3: b = 0
    dotk::givens(2.490569770109506, 0., cosine, sine);
    EXPECT_NEAR(1., cosine, tol);
    EXPECT_NEAR(0., sine, tol);
}

TEST(DOTk_MATH, frobeniusNorm)
{
    size_t nrows = 4;
    size_t ncols = 2;
    std::vector<std::vector<Real> > matrix(nrows, std::vector<Real>(ncols, 0.));
    matrix[0][0] = 1.;
    matrix[1][0] = 2.;
    matrix[2][0] = 3.;
    matrix[3][0] = 4.;
    matrix[0][1] = 5.;
    matrix[1][1] = 6.;
    matrix[2][1] = 7.;
    matrix[3][1] = 8.;

    const Real tolerance = 1e-8;
    EXPECT_NEAR(14.2828568570857, dotk::frobeniusNorm(matrix), tolerance);
}

TEST(DOTk_GTOOLS, didDataChanged)
{
    std::tr1::shared_ptr<dotk::Vector<Real> > vecA = dotk::gtest::allocateControl();
    vecA->fill(2);
    std::tr1::shared_ptr<dotk::Vector<Real> > vecB = dotk::gtest::allocateControl();
    vecB->fill(2);
    EXPECT_FALSE(dotk::gtools::didDataChanged(vecA, vecB));

    vecA->fill(3);
    EXPECT_TRUE(dotk::gtools::didDataChanged(vecA, vecB));
}

}
