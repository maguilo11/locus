/*
 * DOTk_DaiLiaoTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_DaiLiao.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkDaiLiaoTest
{

TEST(DOTk_DaiLiao, setAndGetConstant)
{
    dotk::DOTk_DaiLiao dir;
    EXPECT_EQ(dotk::types::DAI_LIAO_NLCG, dir.getNonlinearCGType());
    Real tol = 1e-8;
    EXPECT_NEAR(0.1, dir.getConstant(), tol);
    dir.setConstant(0.2);
    EXPECT_NEAR(0.2, dir.getConstant(), tol);
}

TEST(DOTk_DaiLiao, computeScaleFactor)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
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

    dotk::DOTk_DaiLiao dir;
    EXPECT_EQ(dotk::types::DAI_LIAO_NLCG, dir.getNonlinearCGType());
    Real tol = 1e-8;
    Real value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getOldPrimal(),
                                        mng.getNewPrimal(),
                                        mng.getTrialStep());
    EXPECT_NEAR(-20.389285714286, value, tol);
}

TEST(DOTk_DaiLiao, getDirection)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
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

    dotk::DOTk_DaiLiao dir;
    EXPECT_EQ(dotk::types::DAI_LIAO_NLCG, dir.getNonlinearCGType());
    dir.getDirection(mng.getOldGradient(),
                     mng.getNewGradient(),
                     mng.getOldPrimal(),
                     mng.getNewPrimal(),
                     mng.getTrialStep());

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 31.3892857142857;
    (*gold)[1] = 18.7785714285714;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(-20.389285714286, dir.getScaleFactor(), tol);
}

TEST(DOTk_DaiLiao, direction)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

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

    dotk::DOTk_DaiLiao dir;
    EXPECT_EQ(dotk::types::DAI_LIAO_NLCG, dir.getNonlinearCGType());
    dir.direction(mng);

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 31.3892857142857;
    (*gold)[1] = 18.7785714285714;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(-20.389285714286, dir.getScaleFactor(), tol);
}

}

