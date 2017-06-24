/*
 * DOTk_DaiYuanHybridTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_DaiYuanHybrid.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkDaiYuanHybridTest
{

TEST(DOTk_DaiYuanHybrid, setAndGetWolfeConstant)
{
    dotk::DOTk_DaiYuanHybrid dir;
    EXPECT_EQ(dotk::types::DAI_YUAN_HYBRID_NLCG, dir.getNonlinearCGType());
    Real tol = 1e-8;
    Real gold = 1. / 3.;
    EXPECT_NEAR(gold, dir.getWolfeConstant(), tol);
    dir.setWolfeConstant(0.5);
    EXPECT_NEAR(0.5, dir.getWolfeConstant(), tol);
}

TEST(DOTk_DaiYuanHybrid, computeScaleFactor)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
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

    dotk::DOTk_DaiYuanHybrid dir;
    EXPECT_EQ(dotk::types::DAI_YUAN_HYBRID_NLCG, dir.getNonlinearCGType());
    // TEST 1: Scaled Dai-Yuan scale
    Real tol = 1e-8;
    Real value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getTrialStep());
    EXPECT_NEAR(10.803571428571, value, tol);
    // TEST 2: Dai-Yuan scale
    dir.setWolfeConstant(1e-3);
    (*primal->control())[0] = 11.;
    (*primal->control())[1] = 22.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -12.;
    (*primal->control())[1] = -23.;
    mng.setNewGradient(*primal->control());
    value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getTrialStep());
    EXPECT_NEAR(5.9557522123894, value, tol);
}

TEST(DOTk_DaiYuanHybrid, getDirection)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
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

    // TEST 1: Scaled Dai-Yuan scale
    dotk::DOTk_DaiYuanHybrid dir;
    EXPECT_EQ(dotk::types::DAI_YUAN_HYBRID_NLCG, dir.getNonlinearCGType());
    dir.getDirection(mng.getOldGradient(), mng.getNewGradient(), mng.getTrialStep());
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = 0.19642857142857;
    (*gold)[1] = -43.607142857142;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(10.803571428571, dir.getScaleFactor(), tol);

    // TEST 2: Dai-Yuan scale
    (*primal->control())[0] = 11.;
    (*primal->control())[1] = 22.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -12.;
    (*primal->control())[1] = -23.;
    mng.setNewGradient(*primal->control());
    dir.getDirection(mng.getOldGradient(), mng.getNewGradient(), mng.getTrialStep());
    (*gold)[0] = 12.067522825323;
    (*gold)[1] = 8.009932778168;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(0.34375256528361, dir.getScaleFactor(), tol);
}

TEST(DOTk_DaiYuanHybrid, direction)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng->setTrialStep(*primal->control());


    // TEST 1: Scaled Dai-Yuan scale
    dotk::DOTk_DaiYuanHybrid dir;
    EXPECT_EQ(dotk::types::DAI_YUAN_HYBRID_NLCG, dir.getNonlinearCGType());
    dir.direction(mng);
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = 0.19642857142857;
    (*gold)[1] = -43.607142857142;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(10.803571428571, dir.getScaleFactor(), tol);

    // TEST 2: Dai-Yuan scale
    (*primal->control())[0] = 11.;
    (*primal->control())[1] = 22.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -12.;
    (*primal->control())[1] = -23.;
    mng->setNewGradient(*primal->control());
    dir.direction(mng);

    (*gold)[0] = 12.067522825323;
    (*gold)[1] = 8.009932778168;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(0.34375256528361, dir.getScaleFactor(), tol);
}

}
