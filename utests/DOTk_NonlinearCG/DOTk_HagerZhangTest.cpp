/*
 * DOTk_HagerZhangTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_HagerZhang.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkHagerZhangTest
{

TEST(DOTk_HagerZhang, setAndGetLowerBoundLimit)
{
    dotk::DOTk_HagerZhang dir;
    EXPECT_EQ(dotk::types::HAGER_ZHANG_NLCG, dir.getNonlinearCGType());
    Real tol = 1e-8;
    EXPECT_NEAR(0.1, dir.getLowerBoundLimit(), tol);
    dir.setLowerBoundLimit(0.4);
    EXPECT_NEAR(0.4, dir.getLowerBoundLimit(), tol);
}

TEST(DOTk_HagerZhang, computeScaleFactor)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
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

    dotk::DOTk_HagerZhang dir;
    EXPECT_EQ(dotk::types::HAGER_ZHANG_NLCG, dir.getNonlinearCGType());
    // TEST 1: Hager Zhang scale
    Real tol = 1e-8;
    Real value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getTrialStep());
    EXPECT_NEAR(25.36734693877, value, tol);

    // TEST 2: Lower bound scale
    primal->control()->fill(1e-1);
    mng.setTrialStep(*primal->control());
    value = dir.computeScaleFactor(mng.getOldGradient(),
                                   mng.getNewGradient(),
                                   mng.getTrialStep());
    EXPECT_NEAR(-70.71067811865, value, tol);
}

TEST(DOTk_HagerZhang, getDirection)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
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

    dotk::DOTk_HagerZhang dir;
    EXPECT_EQ(dotk::types::HAGER_ZHANG_NLCG, dir.getNonlinearCGType());
    // TEST 1: Hager Zhang scale
    dir.getDirection(mng.getOldGradient(), mng.getNewGradient(), mng.getTrialStep());

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -14.367346938775;
    (*gold)[1] = -72.734693877551;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(25.36734693877, dir.getScaleFactor(), tol);

    // TEST 2: Lower bound scale
    primal->control()->fill(1e-1);
    mng.setTrialStep(*primal->control());
    dir.getDirection(mng.getOldGradient(), mng.getNewGradient(), mng.getTrialStep());
    (*gold)[0] = 3.92893218813452;
    (*gold)[1] = -29.07106781186547;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(-70.71067811865, dir.getScaleFactor(), tol);
}

TEST(DOTk_HagerZhang, direction)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
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

    dotk::DOTk_HagerZhang dir;
    EXPECT_EQ(dotk::types::HAGER_ZHANG_NLCG, dir.getNonlinearCGType());
    // TEST 1: Hager Zhang scale
    dir.direction(mng);

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -14.367346938775;
    (*gold)[1] = -72.734693877551;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(25.36734693877, dir.getScaleFactor(), tol);

    // TEST 2: Lower bound scale
    primal->control()->fill(1e-1);
    mng->setTrialStep(*primal->control());
    dir.direction(mng);
    (*gold)[0] =   3.92893218813452;
    (*gold)[1] = -29.07106781186547;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(-70.71067811865, dir.getScaleFactor(), tol);
}

}
