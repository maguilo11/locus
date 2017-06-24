/*
 * DOTk_FirstOrderAlgorithmsFreudensteinRothTest.cpp
 *
 *  Created on: May 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_NonlinearCG.hpp"
#include "DOTk_LineSearchStep.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearchQuasiNewton.hpp"
#include "DOTk_FreudensteinRothObjective.hpp"

namespace DOTkFirstOrderAlgorithmsFreudensteinRothTest
{

TEST(FirstOrderAlgorithmsFreudensteinRoth, HestenesStiefel_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHestenesStiefelNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(16u, nlcg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, PolakRibiere_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPolakRibiereNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 5.;
    (*primal->control())[1] = 4.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(16, nlcg.getNumItrDone());
    EXPECT_NEAR(0, mng->getNewObjectiveFunctionValue(), 1e-8);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, ConjugateDescent_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setConjugateDescentNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(96, nlcg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, HagerZhang_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setHagerZhangNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(550, nlcg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, DaiLiao_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiLiaoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(21, nlcg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, DaiYuan_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(218, nlcg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, DaiYuanHybrid_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setDaiYuanHybridNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(87, nlcg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, PerryShanno_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setPerryShannoNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(35, nlcg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, LiuStorey_UsrDefGrad_CubicIntrpLS)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_NonlinearCG nlcg(step, mng);
    nlcg.setLiuStoreyNlcg();
    nlcg.getMin();

    (*primal->control())[0] = 5.;
    (*primal->control())[1] = 4.;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-6);
    EXPECT_EQ(13, nlcg.getNumItrDone());
    EXPECT_NEAR(0, mng->getNewObjectiveFunctionValue(), 1e-8);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, LBFGS_CubicIntrpLS_)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLbfgsSecantMethod(8);
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(11, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, LDFP_CubicIntrpLS_)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLdfpSecantMethod(8);
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(25, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, LSR1_CubicIntrpLS_)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setCubicLineSearch(primal, 0.25);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setLsr1SecantMethod();
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(10, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, BFGS_CubicIntrpLS_)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.25);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBfgsSecantMethod();
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(21, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, BB_CubicIntrpLS_)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setBarzilaiBorweinSecantMethod();
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(21, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

TEST(FirstOrderAlgorithmsFreudensteinRoth, Sr1_CubicIntrpLS_)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    (*primal->control())[0] = 0.5;
    (*primal->control())[1] = -2.0;
    std::shared_ptr<dotk::DOTk_LineSearchStep> step = std::make_shared<dotk::DOTk_LineSearchStep>(primal);
    std::shared_ptr<dotk::DOTk_FreudensteinRothObjective> objective = std::make_shared<dotk::DOTk_FreudensteinRothObjective>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    mng->setUserDefinedGradient();
    step->setContractionFactor(0.75);
    dotk::DOTk_LineSearchQuasiNewton alg(step, mng);
    alg.setSr1SecantMethod();
    alg.getMin();

    (*primal->control())[0] = 11.41277900335;
    (*primal->control())[1] = -0.89680525194;
    dotk::gtest::checkResults(*mng->getNewPrimal(), *primal->control(), 1e-7);
    EXPECT_EQ(21, alg.getNumItrDone());
    EXPECT_NEAR(48.9843, mng->getNewObjectiveFunctionValue(), 1e-4);
}

}
