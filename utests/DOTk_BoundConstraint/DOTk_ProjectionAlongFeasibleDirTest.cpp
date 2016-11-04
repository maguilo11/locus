/*
 * DOTk_ProjectionAlongFeasibleDirTest.cpp
 *
 *  Created on: Sep 19, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_ArmijoLineSearch.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_ProjectionAlongFeasibleDir.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkProjectionAlongFeasibleDirTest
{

TEST(DOTk_ProjectionAlongFeasibleDir, getDirection)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(1);
    primal->setControlUpperBound(4);
    std::tr1::shared_ptr<dotk::vector<Real> > dir = primal->control()->clone();
    dir->fill(7.);

    // TEST 1: DIRECTION IS NOT FEASIBLE, PROJECT DIRECTION
    dotk::DOTk_ProjectionAlongFeasibleDir bound(primal);
    EXPECT_EQ(dotk::types::PROJECTION_ALONG_FEASIBLE_DIR, bound.type());
    bound.getDirection(primal->control(), dir);
    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    gold->fill(2);
    dotk::gtest::checkResults(*dir, *gold);

    // TEST 2: DIRECTION IS FEASIBLE, DO NOT PROJECT DIRECTION
    dir->fill(1.);
    bound.getDirection(primal->control(), dir);
    gold->fill(1.);
    dotk::gtest::checkResults(*dir, *gold);
}

TEST(DOTk_ProjectionAlongFeasibleDir, constraint)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(1);
    primal->setControlUpperBound(4);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ArmijoLineSearch> step(new dotk::DOTk_ArmijoLineSearch(primal->control()));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    primal->control()->fill(7.);
    mng->setTrialStep(*primal->control());

    dotk::DOTk_ProjectionAlongFeasibleDir bound(primal);
    EXPECT_EQ(dotk::types::PROJECTION_ALONG_FEASIBLE_DIR, bound.type());
    bound.setStepType(dotk::types::bound_step_t::ARMIJO_STEP);

    // TEST 1: DIRECTION IS NOT FEASIBLE, SCALE AND PROJECT DIRECTION
    bound.constraint(step, mng);
    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    gold->fill(5.2154064178466797e-08);
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);

    // TEST 2: DIRECTION IS FEASIBLE, DO NOT PROJECT DIRECTION
    primal->control()->fill(1.);
    mng->setTrialStep(*primal->control());
    bound.constraint(step, mng);
    gold->fill(0.99999999254942);
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
}

}
