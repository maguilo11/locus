/*
 * DOTk_LiuStoreyTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LiuStorey.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLiuStoreyTest
{

TEST(DOTk_LiuStorey, computeScaleFactor)
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
    dotk::DOTk_LiuStorey dir;
    EXPECT_EQ(dotk::types::LIU_STOREY_NLCG, dir.getNonlinearCGType());
    // TEST 1: true scale factor
    Real tol = 1e-8;
    Real value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getTrialStep());
    EXPECT_NEAR(114.4, value, tol);
}

TEST(DOTk_LiuStorey, getDirection)
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

    dotk::DOTk_LiuStorey dir;
    EXPECT_EQ(dotk::types::LIU_STOREY_NLCG, dir.getNonlinearCGType());
    dir.getDirection(mng.getOldGradient(),
                     mng.getNewGradient(),
                     mng.getTrialStep());

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -103.4;
    (*gold)[1] = -250.8;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(114.4, dir.getScaleFactor(), tol);
}

TEST(DOTk_LiuStorey, direction)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng->setTrialStep(*primal->control());

    dotk::DOTk_LiuStorey dir;
    EXPECT_EQ(dotk::types::LIU_STOREY_NLCG, dir.getNonlinearCGType());
    dir.direction(mng);

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -103.4;
    (*gold)[1] = -250.8;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(114.4, dir.getScaleFactor(), tol);
}

}
