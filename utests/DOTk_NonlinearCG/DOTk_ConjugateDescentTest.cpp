/*
 * DOTk_ConjugateDescentTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_ConjugateDescent.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkConjugateDescentTest
{

TEST(DOTk_ConjugateDescent, computeScaleFactor)
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

    dotk::DOTk_ConjugateDescent dir;
    EXPECT_EQ(dotk::types::CONJUGATE_DESCENT_NLCG, dir.getNonlinearCGType());
    Real value = dir.computeScaleFactor(mng.getOldGradient(),
                                        mng.getNewGradient(),
                                        mng.getTrialStep());
    Real tol = 1e-8;
    EXPECT_NEAR(121, value, tol);
}

TEST(DOTk_ConjugateDescent, getDirection)
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

    dotk::DOTk_ConjugateDescent dir;
    EXPECT_EQ(dotk::types::CONJUGATE_DESCENT_NLCG, dir.getNonlinearCGType());
    dir.getDirection(mng.getOldGradient(), mng.getNewGradient(), mng.getTrialStep());

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -110.;
    (*gold)[1] = -264.;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    Real tol = 1e-8;
    EXPECT_NEAR(121, dir.getScaleFactor(), tol);
}

TEST(DOTk_ConjugateDescent, direction)
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

    dotk::DOTk_ConjugateDescent dir;
    EXPECT_EQ(dotk::types::CONJUGATE_DESCENT_NLCG, dir.getNonlinearCGType());
    dir.direction(mng);

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -110.;
    (*gold)[1] = -264.;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    Real tol = 1e-8;
    EXPECT_NEAR(121, dir.getScaleFactor(), tol);
}

}
