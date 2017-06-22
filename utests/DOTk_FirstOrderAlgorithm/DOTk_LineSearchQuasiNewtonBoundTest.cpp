/*
 * DOTk_LineSearchQuasiNewtonBoundTest.cpp
 *
 *  Created on: Sep 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_LineSearchQuasiNewton.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"

namespace DOTk_LineSearchQuasiNewtonBoundTest
{

TEST(LineSearchQuasiNewtonBound, LBFGArmijoLS_ProjectFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);

    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLbfgsSecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LBFGS_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(27, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewtonBound, LDFPArmijoLS_ProjectFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);

    size_t secant_storage = 3;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLdfpSecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LDFP_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::TRIAL_STEP_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(331, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewtonBound, LSR1ArmijoLS_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);

    size_t secant_storage = 2;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLsr1SecantMethod(secant_storage);
    alg.setLsr1SecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LSR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(27, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewtonBound, LSR1ArmijoLS_ProjectFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);

    size_t secant_storage = 3;
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLsr1SecantMethod(secant_storage);
    EXPECT_EQ(dotk::types::LSR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(30, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewtonBound, SR1ArmijoLS_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);

    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setSr1SecantMethod();
    EXPECT_EQ(dotk::types::SR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(59, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewtonBound, SR1ArmijoLS_ProjectFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);

    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setSr1SecantMethod();
    EXPECT_EQ(dotk::types::SR1_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(59, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewtonBound, BBArmijoLS_FeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);

    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBarzilaiBorweinSecantMethod();
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(59, alg.getNumItrDone());
}

TEST(LineSearchQuasiNewtonBound, BBArmijoLS_ProjectFeasibleDirBound)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);

    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBarzilaiBorweinSecantMethod();
    EXPECT_EQ(dotk::types::BARZILAIBORWEIN_INV_HESS, alg.getInvHessianPtr()->getInvHessianType());
    alg.getMin();

    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateControl();
    gold->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold, 1e-6);
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, alg.getStoppingCriterion());
    EXPECT_EQ(59, alg.getNumItrDone());
}

}
