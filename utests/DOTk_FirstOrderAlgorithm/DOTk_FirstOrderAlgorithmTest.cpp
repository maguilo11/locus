/*
 * DOTk_FirstOrderAlgorithmTest.cpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_FirstOrderAlgorithm.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_RoutinesTypeULP.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_LineSearch.hpp"
#include "DOTk_NonlinearCG.hpp"
#include "DOTk_BoundConstraints.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"

namespace dotk
{

}

namespace DOTkFirstOrderAlgorithmTest
{

TEST(FirstOrderAlgorithm, setAndGetMaxNumItr)
{
    dotk::DOTk_FirstOrderAlgorithm alg;
    EXPECT_EQ(5000u, alg.getMaxNumItr());
    alg.setMaxNumItr(10);
    EXPECT_EQ(10u, alg.getMaxNumItr());
}

TEST(FirstOrderAlgorithm, setAndGetNumItrDone)
{
    dotk::DOTk_FirstOrderAlgorithm alg;
    EXPECT_EQ(0u, alg.getNumItrDone());
    alg.setNumItrDone(11);
    EXPECT_EQ(11u, alg.getNumItrDone());
}

TEST(FirstOrderAlgorithm, setAndGetObjectiveFuncTol)
{
    dotk::DOTk_FirstOrderAlgorithm alg;
    Real tol = 1e-8;
    EXPECT_NEAR(5.e-12, alg.getObjectiveFuncTol(), tol);
    alg.setObjectiveFuncTol(0.12);
    EXPECT_NEAR(0.12, alg.getObjectiveFuncTol(), tol);
}

TEST(FirstOrderAlgorithm, setAndGetGradientTol)
{
    dotk::DOTk_FirstOrderAlgorithm alg;
    Real tol = 1e-8;
    EXPECT_NEAR(1.e-8, alg.getGradientTol(), tol);
    alg.setGradientTol(0.123);
    EXPECT_NEAR(0.123, alg.getGradientTol(), tol);
}

TEST(FirstOrderAlgorithm, setAndGetTrialStepTol)
{
    dotk::DOTk_FirstOrderAlgorithm alg;
    Real tol = 1e-8;
    EXPECT_NEAR(1.e-12, alg.getTrialStepTol(), tol);
    alg.setTrialStepTol(0.0123);
    EXPECT_NEAR(0.0123, alg.getTrialStepTol(), tol);
}

TEST(FirstOrderAlgorithm, resetCurrentStateToFormer)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock());
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldPrimal(*primal->control());
    mng->setOldObjectiveFunctionValue(mng->getRoutinesMng()->objective(mng->getOldPrimal()));
    (*primal->control())[0] = 2.;
    (*primal->control())[1] = 3.;
    mng->setNewPrimal(*primal->control());
    mng->setNewObjectiveFunctionValue(mng->getRoutinesMng()->objective(mng->getNewPrimal()));
    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    dotk::DOTk_FirstOrderAlgorithm alg;
    alg.resetCurrentStateToFormer(mng);

    Real tol = 1e-8;
    EXPECT_NEAR(mng->getNewObjectiveFunctionValue(), mng->getOldObjectiveFunctionValue(), tol);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());
}

TEST(FirstOrderAlgorithm, setAndGetAlgorithmType)
{
    dotk::DOTk_FirstOrderAlgorithm alg;
    EXPECT_EQ(dotk::types::DOTk_ALGORITHM_DISABLED, alg.getAlgorithmType());
    alg.setAlgorithmType(dotk::types::LINE_SEARCH_INEXACT_NEWTON);
    EXPECT_EQ(dotk::types::LINE_SEARCH_INEXACT_NEWTON, alg.getAlgorithmType());
}

TEST(FirstOrderAlgorithm, setAndGetOptProbStoppingCriterion)
{
    dotk::DOTk_FirstOrderAlgorithm alg;
    EXPECT_EQ(dotk::types::OPT_ALG_HAS_NOT_CONVERGED, alg.getStoppingCriterion());
    alg.setStoppingCriterion(dotk::types::NaN_TRIAL_STEP_NORM);
    EXPECT_EQ(dotk::types::NaN_TRIAL_STEP_NORM, alg.getStoppingCriterion());
}

TEST(FirstOrderAlgorithm, checkStoppingCriteria)
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
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    dotk::DOTk_FirstOrderAlgorithm alg;

    Real tol = 1e-8;
    // TEST 1: ALG. HAS NOT CONVERGED
    EXPECT_FALSE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::OPT_ALG_HAS_NOT_CONVERGED, alg.getStoppingCriterion());

    // TEST 2: NaN TRIAL STEP NORM
    alg.setNumItrDone(1);
    (*primal->control())[0] = std::numeric_limits<Real>::quiet_NaN();
    (*primal->control())[1] = std::numeric_limits<Real>::quiet_NaN();
    mng->setTrialStep(*primal->control());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::NaN_TRIAL_STEP_NORM, alg.getStoppingCriterion());
    EXPECT_NEAR(mng->getNewObjectiveFunctionValue(), mng->getOldObjectiveFunctionValue(), tol);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());

    // TEST 3: NaN GRADIENT NORM
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2.;
    mng->setTrialStep(*primal->control());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    (*primal->control())[0] = std::numeric_limits<Real>::quiet_NaN();
    (*primal->control())[1] = std::numeric_limits<Real>::quiet_NaN();
    mng->setNewGradient(*primal->control());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::NaN_GRADIENT_NORM, alg.getStoppingCriterion());
    EXPECT_NEAR(mng->getNewObjectiveFunctionValue(), mng->getOldObjectiveFunctionValue(), tol);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());
    dotk::gtest::checkResults(*mng->getNewPrimal(), *mng->getOldPrimal());

    // TEST 4: TRIAL STEP NORM IS LESS THAN TOLERANCE
    (*primal->control())[0] = std::numeric_limits<Real>::min();
    (*primal->control())[1] = std::numeric_limits<Real>::min();
    mng->setTrialStep(*primal->control());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());

    // TEST 5: GRADIENT NORM IS LESS THAN TOLERANCE
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2.;
    mng->setTrialStep(*primal->control());
    mng->setNormTrialStep(mng->getTrialStep()->norm());
    (*primal->control())[0] = std::numeric_limits<Real>::min();
    (*primal->control())[1] = std::numeric_limits<Real>::min();
    mng->setNewGradient(*primal->control());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::GRADIENT_TOL_SATISFIED, alg.getStoppingCriterion());

    // TEST 6: OBJECTIVE FUNCTION VALUE IS LESS THAN TOLERANCE
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    mng->setNormNewGradient(mng->getNewGradient()->norm());
    mng->setNewObjectiveFunctionValue(std::numeric_limits<Real>::min());
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());

    // TEST 7: MAXIMUM NUMBER OF ITERATIONS REACHED
    alg.setNumItrDone(5000);
    mng->setNewObjectiveFunctionValue(1.);
    EXPECT_TRUE(alg.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::MAX_NUM_ITR_REACHED, alg.getStoppingCriterion());
}

}
