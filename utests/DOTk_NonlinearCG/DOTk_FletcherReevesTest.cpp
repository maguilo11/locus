/*
 * DOTk_FletcherReevesTest.cpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_FletcherReeves.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkDescentDirectionTest
{

TEST(DOTk_FletcherReeves, setAndGetMinCosineAngleTol)
{
    dotk::DOTk_FletcherReeves dir;
    Real tol = 1e-8;
    EXPECT_NEAR(1e-2, dir.getMinCosineAngleTol(), tol);
    dir.setMinCosineAngleTol(0.123);
    EXPECT_NEAR(0.123, dir.getMinCosineAngleTol(), tol);
}

TEST(DOTk_FletcherReeves, computeCosineAngle)
{
    std::tr1::shared_ptr<dotk::Vector<Real> > gold_grad = dotk::gtest::allocateControl();
    std::tr1::shared_ptr<dotk::Vector<Real> > trial_step = gold_grad->clone();
    (*gold_grad)[0] = 1.;
    (*gold_grad)[1] = 2.;
    (*trial_step)[0] = 11.;
    (*trial_step)[1] = -22.;

    dotk::DOTk_FletcherReeves dir;
    Real value = dir.computeCosineAngle(gold_grad, trial_step);
    Real tol = 1e-8;
    EXPECT_NEAR(0.6, value, tol);
    (*trial_step)[0] = -1.;
    (*trial_step)[1] = -2.;
    value = dir.computeCosineAngle(gold_grad, trial_step);
    EXPECT_NEAR(1., value, tol);
}

TEST(DOTk_FletcherReeves, steepestDescent)
{
    std::tr1::shared_ptr<dotk::Vector<Real> > grad = dotk::gtest::allocateControl();
    std::tr1::shared_ptr<dotk::Vector<Real> > trial_step = grad->clone();
    (*grad)[0] = 1.;
    (*grad)[1] = 2.;

    dotk::DOTk_FletcherReeves dir;
    dir.steepestDescent(grad, trial_step);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = grad->clone();
    (*gold)[0] = -1;
    (*gold)[1] = -2;
    dotk::gtest::checkResults(*trial_step, *gold);
}

TEST(DOTk_FletcherReeves, isTrialStepOrthogonalToSteepestDescent)
{
    dotk::DOTk_FletcherReeves dir;
    Real cosine_value = 0.6;
    EXPECT_FALSE(dir.isTrialStepOrthogonalToSteepestDescent(cosine_value));
    cosine_value = 0.0099;
    EXPECT_TRUE(dir.isTrialStepOrthogonalToSteepestDescent(cosine_value));
    cosine_value = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_TRUE(dir.isTrialStepOrthogonalToSteepestDescent(cosine_value));
}

TEST(DOTk_FletcherReeves, computeScaleFactor)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng.setNewGradient(*primal->control());

    dotk::DOTk_FletcherReeves dir;
    EXPECT_EQ(dotk::types::FLETCHER_REEVES_NLCG, dir.getNonlinearCGType());
    Real value = dir.computeScaleFactor(mng.getOldGradient(), mng.getNewGradient());
    Real tol = 1e-8;
    EXPECT_NEAR(121., value, tol);
}

TEST(DOTk_FletcherReeves, getDirection)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng.setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng.setTrialStep(*primal->control());

    dotk::DOTk_FletcherReeves dir;
    EXPECT_EQ(dotk::types::FLETCHER_REEVES_NLCG, dir.getNonlinearCGType());
    dir.getDirection(mng.getOldGradient(), mng.getNewGradient(), mng.getTrialStep());

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -110.;
    (*gold)[1] = -264.;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(121., dir.getScaleFactor(), tol);
}

TEST(DOTk_FletcherReeves, direction)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
    mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng->setTrialStep(*primal->control());
    dotk::DOTk_FletcherReeves dir;
    EXPECT_EQ(dotk::types::FLETCHER_REEVES_NLCG, dir.getNonlinearCGType());

    // TEST 1
    dir.direction(mng);
    std::tr1::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -110.;
    (*gold)[1] = -264.;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(121., dir.getScaleFactor(), tol);

    // TEST 2: TAKE STEEPEST DESCENT (NaN cosine(angle) value)
    primal->control()->fill(0.);
    mng->setTrialStep(*primal->control());
    dir.direction(mng);
    (*gold)[0] = 11.;
    (*gold)[1] = -22.;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);

    // TEST 3: TAKE STEEPEST DESCENT (Small cosine(angle) value)
    (*primal->control())[0] = 1e-8;
    (*primal->control())[1] = 2e-11;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = 1.32e4;
    (*primal->control())[1] = -2.23e6;
    mng->setTrialStep(*primal->control());
    dir.direction(mng);
    (*gold)[0] = 11.;
    (*gold)[1] = -22.;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
}

}
