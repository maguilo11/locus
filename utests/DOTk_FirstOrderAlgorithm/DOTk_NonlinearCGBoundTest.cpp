/*
 * DOTk_NonlinearCGBoundTest.cpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include <fstream>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_NonlinearCG.hpp"
#include "DOTk_ProjectedStep.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_ProjectedLineSearchStep.hpp"

namespace DOTkNonlinearCGBoundTest
{

TEST(NonlinearCGBound, ProjectedStep_FletcherReeves_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setFletcherReevesNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(61, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_PolakRibiere_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(121, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_HestenesStiefel_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);
    step->setContractionFactor(0.4);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(33, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_ConjugateDescent_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(135, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_HagerZhang_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(29, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_DaiLiao_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);
    step->setContractionFactor(0.4);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(25, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_DaiYuan_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);
    step->setContractionFactor(0.75);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(28, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_DaiYuanHybrid_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);
    step->setContractionFactor(0.3);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(53, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_PerryShanno_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);
    step->setContractionFactor(0.2);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(47, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ProjectedStep_LiuStorey_CubicLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(-1e1);
    primal->setControlUpperBound(1e1);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedStep> step(new dotk::DOTk_ProjectedStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal);
    step->setContractionFactor(0.2);

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(46, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, HestenesStiefel_UsrDefGrad_ArmijoLS_ProjectionFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.75);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(91, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PolakRibiere_UsrDefGrad_ArmijoLS_ProjectionFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.1);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(1970, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, ConugateDescent_UsrDefGrad_ArmijoLS_ProjectionFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(135, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, HagerZhang_UsrDefGrad_ArmijoLS_ProjectFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(108, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, DaiLiao_UsrDefGrad_ArmijoLS_ProjectFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(25, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, DaiYuanHybrid_UsrDefGrad_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(48, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PerryShanno_UsrDefGrad_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.25);

    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();
    primal->control()->fill(1.);

    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(29, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PerryShanno_GradFD_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(5e-9);
    mng->setForwardFiniteDiffGradient(primal);
    step->setArmijoLineSearch(primal, 0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();
    primal->control()->fill(1.);

    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(41, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PerryShanno_GradPrllFD_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(5e-9);
    mng->setParallelForwardFiniteDiffGradient(primal);
    step->setArmijoLineSearch(primal, 0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();
    primal->control()->fill(1.);

    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(41, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PerryShanno_GradBD_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(5e-9);
    mng->setBackwardFiniteDiffGradient(primal);
    step->setArmijoLineSearch(primal, 0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();
    primal->control()->fill(1.);

    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(39, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PerryShanno_GradPrllBD_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(5e-9);
    mng->setParallelBackwardFiniteDiffGradient(primal);
    step->setArmijoLineSearch(primal, 0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();
    primal->control()->fill(1.);

    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(39, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PerryShanno_GradCD_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(5e-9);
    mng->setCentralFiniteDiffGradient(primal);
    step->setArmijoLineSearch(primal, 0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();
    primal->control()->fill(1.);

    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(29, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, PerryShanno_GradPrllCD_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(5e-9);
    mng->setParallelCentralFiniteDiffGradient(primal);
    step->setArmijoLineSearch(primal, 0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();
    primal->control()->fill(1.);

    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(29, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

TEST(NonlinearCGBound, LiuStorey_UsrDefGrad_ArmijoLS_ProjectedFeasibleDir)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    primal->setControlLowerBound(0);
    primal->setControlUpperBound(5);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_ProjectedLineSearchStep> step(new dotk::DOTk_ProjectedLineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal);
    step->setContractionFactor(0.25);
    step->setProjectionAlongFeasibleDirConstraint(primal);
    step->setArmijoBoundConstraintMethodStep();

    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(141, nlcg.getNumItrDone());
    EXPECT_EQ(dotk::types::OBJECTIVE_FUNC_TOL_SATISFIED, nlcg.getStoppingCriterion());
}

}
