/*
 * DOTk_FeasibleDirectionTest.cpp
 *
 *  Created on: Sep 16, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_ArmijoLineSearch.hpp"
#include "DOTk_FeasibleDirection.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkFeasibleDirectionTest
{

TEST(DOTk_FeasibleDirection, getDirection)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(1);
    primal->setControlUpperBound(4);

    std::shared_ptr<dotk::Vector<Real> > dir = primal->control()->clone();
    dir->fill(7.);

    // TEST 1: NOT FEASIBLE
    dotk::DOTk_FeasibleDirection bound(primal);
    EXPECT_EQ(dotk::types::FEASIBLE_DIR, bound.type());
    bound.getDirection(primal->control(), dir);

    Real tol = 1e-8;
    EXPECT_EQ(3, bound.getNumFeasibleItr());
    EXPECT_NEAR(0.5, bound.getContractionStep(), tol);
    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    gold->fill(1.75);
    dotk::gtest::checkResults(*dir, *gold);

    // TEST 2: FEASIBLE
    bound.getDirection(primal->control(), dir);
    EXPECT_EQ(1, bound.getNumFeasibleItr());
    EXPECT_NEAR(0.5, bound.getContractionStep(), tol);
    dotk::gtest::checkResults(*dir, *gold);
}

TEST(DOTk_FeasibleDirection, constraint)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(1);
    primal->setControlUpperBound(4);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_ArmijoLineSearch> step = std::make_shared<dotk::DOTk_ArmijoLineSearch>(primal->control());
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    mng->setNewPrimal(*primal->control());
    std::shared_ptr<dotk::Vector<Real> > dir = primal->control()->clone();
    dir->fill(7.);
    mng->setTrialStep(*dir);

    // TEST 1: NOT FEASIBLE
    dotk::DOTk_FeasibleDirection bound(primal);
    EXPECT_EQ(dotk::types::FEASIBLE_DIR, bound.type());
    bound.constraint(step, mng);

    Real tol = 1e-8;
    EXPECT_EQ(3, bound.getNumFeasibleItr());
    EXPECT_NEAR(0.5, bound.getContractionStep(), tol);
    primal->control()->fill(1.75);
    dotk::gtest::checkResults(*mng->getTrialStep(), *primal->control());

    // TEST 2: FEASIBLE
    bound.constraint(step, mng);
    EXPECT_EQ(1, bound.getNumFeasibleItr());
    EXPECT_NEAR(0.5, bound.getContractionStep(), tol);
    dotk::gtest::checkResults(*mng->getTrialStep(), *primal->control());
}

}
