/*
 * DOTk_PerryShannoTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_PerryShanno.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkPerryShannoTest
{

TEST(PerryShanno, setAndGetAlphaScaleFactor)
{
    dotk::DOTk_PerryShanno dir;
    EXPECT_EQ(dotk::types::PERRY_SHANNO_NLCG, dir.getNonlinearCGType());
    Real tol = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::min(), dir.getAlphaScaleFactor(), tol);
    dir.setAlphaScaleFactor(0.21);
    EXPECT_NEAR(0.21, dir.getAlphaScaleFactor(), tol);
}

TEST(PerryShanno, setAndGetThetaScaleFactor)
{
    dotk::DOTk_PerryShanno dir;
    EXPECT_EQ(dotk::types::PERRY_SHANNO_NLCG, dir.getNonlinearCGType());
    Real tol = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::min(), dir.getThetaScaleFactor(), tol);
    dir.setThetaScaleFactor(0.31);
    EXPECT_NEAR(0.31, dir.getThetaScaleFactor(), tol);
}

TEST(PerryShanno, setAndGetLowerBoundLimit)
{
    dotk::DOTk_PerryShanno dir;
    EXPECT_EQ(dotk::types::PERRY_SHANNO_NLCG, dir.getNonlinearCGType());
    Real tol = 1e-8;
    EXPECT_NEAR(0.1, dir.getLowerBoundLimit(), tol);
    dir.setLowerBoundLimit(0.41);
    EXPECT_NEAR(0.41, dir.getLowerBoundLimit(), tol);
}

TEST(PerryShanno, computeAlphaScaleFactor)
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

    dotk::DOTk_PerryShanno dir;
    EXPECT_EQ(dotk::types::PERRY_SHANNO_NLCG, dir.getNonlinearCGType());
    Real alpha = dir.computeAlphaScaleFactor(mng.getOldGradient(),
                                             mng.getNewGradient(),
                                             mng.getTrialStep());
    Real tol = 1e-8;
    EXPECT_NEAR(1.1785714285714, alpha, tol);
}

TEST(PerryShanno, computeThetaScaleFactor)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldPrimal(*primal->control());
    (*primal->control())[0] = 2.;
    (*primal->control())[1] = 3.;
    mng.setNewPrimal(*primal->control());
    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng.setNewGradient(*primal->control());

    dotk::DOTk_PerryShanno dir;
    EXPECT_EQ(dotk::types::PERRY_SHANNO_NLCG, dir.getNonlinearCGType());
    Real theta = dir.computeThetaScaleFactor(mng.getOldGradient(),
                                             mng.getNewGradient(),
                                             mng.getOldPrimal(),
                                             mng.getNewPrimal());
    Real tol = 1e-8;
    EXPECT_NEAR(0.01470588235294, theta, tol);
}

TEST(PerryShanno, computeScaleFactor)
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

    // TEST 1: true scale factor
    Real tol = 1e-8;
    dotk::DOTk_PerryShanno dir;
    Real value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getTrialStep());
    EXPECT_NEAR(25.36734693877, value, tol);

    // TEST 2: Lower bound scale factor
    primal->control()->fill(1e-1);
    mng.setTrialStep(*primal->control());
    value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getTrialStep());
    EXPECT_NEAR(-70.71067811865, value, tol);
}

TEST(PerryShanno, getDirection)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldPrimal(*primal->control());
    (*primal->control())[0] = 2.;
    (*primal->control())[1] = 3.;
    mng.setNewPrimal(*primal->control());
    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng.setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng.setTrialStep(*primal->control());

    dotk::DOTk_PerryShanno dir;
    EXPECT_EQ(dotk::types::PERRY_SHANNO_NLCG, dir.getNonlinearCGType());
    dir.getDirection(mng.getOldGradient(),
                     mng.getNewGradient(),
                     mng.getOldPrimal(),
                     mng.getNewPrimal(),
                     mng.getTrialStep());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -0.419267707083;
    (*gold)[1] = -0.722989195678;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(25.36734693877, dir.getScaleFactor(), tol);
    EXPECT_NEAR(1.1785714285714, dir.getAlphaScaleFactor(), tol);
    EXPECT_NEAR(0.01470588235294, dir.getThetaScaleFactor(), tol);
}

TEST(PerryShanno, direction)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
    mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldPrimal(*primal->control());
    (*primal->control())[0] = 2.;
    (*primal->control())[1] = 3.;
    mng->setNewPrimal(*primal->control());
    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng->setTrialStep(*primal->control());

    dotk::DOTk_PerryShanno dir;
    EXPECT_EQ(dotk::types::PERRY_SHANNO_NLCG, dir.getNonlinearCGType());
    dir.direction(mng);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -0.419267707083;
    (*gold)[1] = -0.722989195678;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(25.36734693877, dir.getScaleFactor(), tol);
    EXPECT_NEAR(1.1785714285714, dir.getAlphaScaleFactor(), tol);
    EXPECT_NEAR(0.01470588235294, dir.getThetaScaleFactor(), tol);
}

}
