/*
 * DOTk_NonlinearCGTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_NonlinearCG.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_BealeObjective.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"


namespace DOTkNonlinearCGTest
{

TEST(NonlinearCG, FletcherReeves_UsrDefGrad_GoldenSectionLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(36, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(42, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HestenesStiefel_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(25, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PolakRibiere_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(37, nlcg.getNumItrDone());
}

TEST(NonlinearCG, ConjugateDescent_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(62, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiLiao_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(15, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuanHybrid_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(38, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuan_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(57, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HagerZhang_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(24, nlcg.getNumItrDone());
}

TEST(NonlinearCG, LiuStorey_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(19, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PerryShanno_UsrDefGrad_CubicIntrpLS_BealeObjFunc)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 1);
    std::tr1::shared_ptr<dotk::DOTk_BealeObjective> objective(new dotk::DOTk_BealeObjective);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 0.5;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(23u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, Daniels_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    hessian->setCentralDifference(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDanielsNlcg(hessian);
    nlcg.getMin();

    primal->control()->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(14u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, Daniels_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);

    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    hessian->setCentralDifference(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDanielsNlcg(hessian);
    nlcg.getMin();

    primal->control()->fill(1);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(21u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(118, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(47u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_GradFD_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(1e-7);
    mng->setForwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(40u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_GradPrllFD_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(1e-7);
    mng->setParallelForwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(40u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_GradBD_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(1e-8);
    mng->setBackwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(21u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_GradPrllBD_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(1e-8);
    mng->setParallelBackwardFiniteDiffGradient(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(21u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_GradCD_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(1e-7);
    mng->setCentralFiniteDiffGradient(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(33u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_GradPrllCD_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    primal->control()->fill(1e-7);
    mng->setParallelCentralFiniteDiffGradient(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-6);
    EXPECT_EQ(33u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PolakRibiere_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(34u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PolakRibiere_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(49, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HestenesStiefel_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(119u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HestenesStiefel_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(19, nlcg.getNumItrDone());
}

TEST(NonlinearCG, ConjugateDescent_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(398, nlcg.getNumItrDone());
}

TEST(NonlinearCG, ConjugateDescent_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(93, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HagerZhang_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(46, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HagerZhang_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(19, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiLiao_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(55, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiLiao_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(22, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuan_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(240, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuan_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(26, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuanHybrid_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(44, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuanHybrid_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(23, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PerryShanno_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 5e-7);
    EXPECT_EQ(30, nlcg.getNumItrDone());
}

TEST(NonlinearCG, LiuStorey_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(22u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, Daniels_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    hessian->setSecondOrderForwardDifference(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDanielsNlcg(hessian);
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(41u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng); // TEST DEFAULT SETTINGS
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(75u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PolakRibiere_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(152, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HestenesStiefel_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(35, nlcg.getNumItrDone());
}

TEST(NonlinearCG, ConjugateDescent_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(114, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HagerZhang_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(75, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiLiao_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(25, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuan_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(45, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuanHybrid_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(48, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PerryShanno_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(29, nlcg.getNumItrDone());
}

TEST(NonlinearCG, LiuStorey_UsrDefGrad_ArmijoLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setArmijoLineSearch(primal, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(147u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, Daniels_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));
    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    hessian->setSecondOrderForwardDifference(*primal->control());
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDanielsNlcg(hessian);
    nlcg.getMin();

    primal->control()->fill(1.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(32u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, FletcherReeves_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(52u, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PolakRibiere_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(59, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HestenesStiefel_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(31, nlcg.getNumItrDone());
}

TEST(NonlinearCG, ConjugateDescent_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    step->setContractionFactor(0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(61, nlcg.getNumItrDone());
}

TEST(NonlinearCG, HagerZhang_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(32, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiLiao_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(35, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuan_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(119, nlcg.getNumItrDone());
}

TEST(NonlinearCG, DaiYuanHybrid_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal, 0.9, 0.25);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(56, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PerryShanno_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(26, nlcg.getNumItrDone());
}

TEST(NonlinearCG, PerryShanno_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal, 0.9, 0.7);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(93, nlcg.getNumItrDone());
}

TEST(NonlinearCG, LiuStorey_UsrDefGrad_GoldsteinLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldsteinLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(45, nlcg.getNumItrDone());
}

TEST(NonlinearCG, LiuStorey_UsrDefGrad_GoldenSectionLS)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchStep> step(new dotk::DOTk_LineSearchStep(primal));
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    mng->setUserDefinedGradient();
    step->setGoldenSectionLineSearch(primal);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 1.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-5);
    EXPECT_EQ(26, nlcg.getNumItrDone());
}

}
