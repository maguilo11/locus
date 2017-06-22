/*
 * DOTk_InexactTrustRegionSQPTest.cpp
 *
 *  Created on: Dec 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_AugmentedSystem.hpp"
#include "DOTk_AssemblyManager.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_InexactTrustRegionSQP.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_NocedalAndWrightEquality.hpp"
#include "DOTk_NocedalAndWrightObjective.hpp"
#include "DOTk_UserDefinedHessianTypeCNP.hpp"
#include "DOTk_InexactTrustRegionSqpSolverMng.hpp"

namespace DOTkInexactTrustRegionSQPTest
{

void setSqpTestInitialGuess(std::shared_ptr<dotk::DOTk_Primal> & primal_);
void setSqpTestData(const std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);

TEST(DOTk_InexactTrustRegionSQPTest, AugmentedSystem)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    std::shared_ptr<dotk::Vector<Real> > input(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    input->fill(1);
    std::shared_ptr<dotk::Vector<Real> > output = input->clone();
    std::shared_ptr<dotk::DOTk_AugmentedSystem> augmented_system(new dotk::DOTk_AugmentedSystem);
    augmented_system->apply(mng, input, output);

    // RESULTS
    (*primal->dual())[0] = 0.4;
    (*primal->dual())[1] = 11.6;
    (*primal->dual())[2] = 18.39;
    dotk::gtest::checkResults(*output->dual(), *primal->dual());
    (*primal->control())[0] = 7.12;
    (*primal->control())[1] = 14.97;
    (*primal->control())[2] = 6.5;
    (*primal->control())[3] = 3.4;
    (*primal->control())[4] = 3.4;
    dotk::gtest::checkResults(*output->control(), *primal->control());
}

TEST(DOTk_InexactTrustRegionSQPTest, solveDualProb)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    (*mng->getNewGradient())[0] = -0.737271611192309;
    (*mng->getNewGradient())[1] = -0.755262411678706;
    (*mng->getNewGradient())[2] = -0.0474142630809581;
    (*mng->getNewGradient())[3] = 0.112608874817276;
    (*mng->getNewGradient())[4] = 0.112608874817276;

    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        sqp_solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    sqp_solver_mng->setDefaultKrylovSolvers(primal, hessian);

    dotk::types::solver_stop_criterion_t criterion = sqp_solver_mng->solveDualProb(mng);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->dual()->clone();
    (*gold)[0] = 0.0207961931305597;
    (*gold)[1] = -0.019798867645168;
    (*gold)[2] = 0.0834391241457092;
    EXPECT_EQ(dotk::types::SOLVER_TOLERANCE_SATISFIED, criterion);
    dotk::gtest::checkResults(*mng->m_DeltaDual, *gold);
}

TEST(DOTk_InexactTrustRegionSQPTest, computeTangentialStep)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    (*mng->getProjectedTangentialStep())[0] = 0.0196000125684081;
    (*mng->getProjectedTangentialStep())[1] = - 0.0219737165126789;
    (*mng->getProjectedTangentialStep())[2] = 0.0361541754328155;
    (*mng->getProjectedTangentialStep())[3] = - 0.00246400460771204;
    (*mng->getProjectedTangentialStep())[4] = - 0.00246400460771204;

    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        sqp_solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    sqp_solver_mng->setDefaultKrylovSolvers(primal, hessian);

    dotk::types::solver_stop_criterion_t criterion = sqp_solver_mng->solveTangentialProb(mng);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.019600012568408;
    (*gold)[1] = - 0.0219737165126789;
    (*gold)[2] = 0.0361541754328155;
    (*gold)[3] = - 0.00246400460771204;
    (*gold)[4] = - 0.00246400460771204;
    EXPECT_EQ(dotk::types::NEGATIVE_CURVATURE_DETECTED, criterion);
    dotk::gtest::checkResults(*mng->getTangentialStep(), *gold);
}

TEST(DOTk_InexactTrustRegionSQPTest, computeQuasiNormalStep)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    mng->getRoutinesMng()->equalityConstraint(mng->getNewPrimal(), mng->getNewEqualityConstraintResidual());

    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        sqp_solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    sqp_solver_mng->setDefaultKrylovSolvers(primal, hessian);

    // TEST 1: TAKE FULL QUASI-NORMAL STEP
    dotk::types::solver_stop_criterion_t criterion = sqp_solver_mng->solveQuasiNormalProb(mng);
    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.073897884488436552;
    (*gold)[1] = -0.087999993220813047;
    (*gold)[2] = -0.088747496365614562;
    (*gold)[3] = 0.03520565983020861;
    (*gold)[4] = 0.03520565983020861;
    EXPECT_EQ(dotk::types::NEGATIVE_CURVATURE_DETECTED, criterion);
    dotk::gtest::checkResults(*mng->getNormalStep(), *gold);

    // TEST 2: TAKE NORMAL CAUCHY STEP
    mng->setTrustRegionRadius(0.1);
    criterion = sqp_solver_mng->solveQuasiNormalProb(mng);
    (*gold)[0] = 0.033957683974826534;
    (*gold)[1] = -0.049762295553151051;
    (*gold)[2] = -0.046227597157793274;
    (*gold)[3] = 0.017798860937759994;
    (*gold)[4] = 0.017798860937759994;
    EXPECT_EQ(dotk::types::TRUST_REGION_VIOLATED, criterion);
    dotk::gtest::checkResults(*mng->getNormalStep(), *gold);
    dotk::gtest::checkResults(*mng->getNormalCauchyStep(), *gold);

    // TEST 3: TAKE SCALED QUASI NORMAL STEP
    mng->setTrustRegionRadius(0.19125);
    criterion = sqp_solver_mng->solveQuasiNormalProb(mng);
    (*gold)[0] = 0.071113489135467028;
    (*gold)[1] = -0.08998566316274223;
    (*gold)[2] = -0.088500038969565112;
    (*gold)[3] = 0.034800827508313426;
    (*gold)[4] = 0.034800827508313426;
    EXPECT_EQ(dotk::types::NEGATIVE_CURVATURE_DETECTED, criterion);
    dotk::gtest::checkResults(*mng->getNormalStep(), *gold);
}

TEST(DOTk_InexactTrustRegionSQPTest, solveTangentialSubProb)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    (*primal->dual())[0] = 0.0207961931305597;
    (*primal->dual())[1] = -0.019798867645168;
    (*primal->dual())[2] = 0.0834391241457092;
    mng->getNewDual()->update(1., *primal->dual(), 0.);
    mng->getNewPrimal()->control()->update(1., *primal->control(), 0.);
    (*primal->control())[0] = -0.737271611192309;
    (*primal->control())[1] = -0.755262411678706;
    (*primal->control())[2] = -0.0474142630809581;
    (*primal->control())[3] = 0.112608874817276;
    (*primal->control())[4] = 0.112608874817276;
    mng->getNewGradient()->control()->update(1., *primal->control(), 0.);
    (*primal->control())[0] = 0.821886987715101;
    (*primal->control())[1] = 0.661914924684898;
    (*primal->control())[2] = -0.0339371478280196;
    (*primal->control())[3] = 0.0810737062636601;
    (*primal->control())[4] = 0.0810737062636601;
    mng->getHessTimesNormalStep()->control()->update(1., *primal->control(), 0.);

    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        sqp_solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    sqp_solver_mng->setDefaultKrylovSolvers(primal, hessian);

    sqp_solver_mng->solveTangentialSubProb(mng);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.0196000125684081;
    (*gold)[1] = -0.0219737165126789;
    (*gold)[2] = 0.0361541754328155;
    (*gold)[3] = -0.00246400460771204;
    (*gold)[4] = -0.00246400460771204;
    dotk::gtest::checkResults(*mng->getProjectedTangentialStep()->control(), *gold);
    dotk::gtest::checkResults(*mng->getProjTangentialCauchyStep()->control(), *gold);
    (*gold)[0] = -0.00166757862663005;
    (*gold)[1] = 0.00186953451566829;
    (*gold)[2] = -0.00307601487523399;
    (*gold)[3] = 0.000209638713516003;
    (*gold)[4] = 0.000209638713516003;
    dotk::gtest::checkResults(*mng->getProjectedGradient()->control(), *gold);
}

TEST(DOTk_InexactTrustRegionSQPTest, apply)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    mng->getTrialStep()->fill(1);
    dotk::DOTk_UserDefinedHessianTypeCNP hessian;
    hessian.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    std::shared_ptr<dotk::Vector<Real> > gold = mng->getMatrixTimesVector()->clone();
    (*gold)[0] = -186.38387000271706;
    (*gold)[1] = -147.31620187919;
    (*gold)[2] = 2.7745413983048834;
    (*gold)[3] = -2.2645068986013062;
    (*gold)[4] = -2.2645068986013062;
    dotk::gtest::checkResults(*mng->getMatrixTimesVector(), *gold);
}

TEST(DOTk_InexactTrustRegionSQPTest, storePreviousSolution)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    hessian->setFullSpaceHessian();
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);

    mng->setNewObjectiveFunctionValue(0.1);
    mng->setOldObjectiveFunctionValue(0.2);
    mng->getNewPrimal()->fill(4);
    mng->getOldPrimal()->fill(2);
    mng->getNewGradient()->fill(2);
    mng->getOldGradient()->fill(3);
    mng->getNewEqualityConstraintResidual()->fill(5.);
    mng->getOldEqualityConstraintResidual()->fill(10.);

    sqp.storePreviousSolution(mng);

    Real tolerance = 1e-8;
    EXPECT_NEAR(0.1, mng->getOldObjectiveFunctionValue(), tolerance);
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateData(5, 4.);
    dotk::gtest::checkResults(*mng->getOldPrimal(), *gold);
    gold->fill(2.);
    dotk::gtest::checkResults(*mng->getOldGradient(), *gold);
    std::shared_ptr<dotk::Vector<Real> > eq_constraint_gold = dotk::gtest::allocateData(3, 5.);
    dotk::gtest::checkResults(*mng->getOldEqualityConstraintResidual(), *eq_constraint_gold);
}

TEST(DOTk_InexactTrustRegionSQPTest, checkConvergence)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);

    // TEST 1: SMALL TRUST REGION
    mng->getNewGradient()->fill(1);
    mng->getNewEqualityConstraintResidual()->fill(1.);
    mng->getTrialStep()->fill(1);
    mng->setTrustRegionRadius(1e-11);
    EXPECT_TRUE(sqp.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::TRUST_REGION_RADIUS_SMALLER_THAN_TRIAL_STEP_NORM, sqp.getStoppingCriterion());

    // TEST 2: MAX NUMBER OF ITERATIONS REACHED
    sqp.setNumItrDone(51);
    EXPECT_TRUE(sqp.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::MAX_NUM_ITR_REACHED, sqp.getStoppingCriterion());

    // TEST 3: OPTIMALITY AND FEASIBILITY TOLERANCES SATISFIED
    sqp.setNumItrDone(1);
    mng->getNewGradient()->fill(1e-13);
    mng->getNewEqualityConstraintResidual()->fill(1e-13);
    EXPECT_TRUE(sqp.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::OPTIMALITY_AND_FEASIBILITY_SATISFIED, sqp.getStoppingCriterion());

    // TEST 4: NaN GRADIENT NORM
    sqp.setNumItrDone(1);
    mng->setTrustRegionRadius(1);
    mng->getOldPrimal()->fill(3);
    mng->getNewGradient()->fill(std::numeric_limits<Real>::quiet_NaN());
    mng->getOldGradient()->fill(1);
    mng->getNewEqualityConstraintResidual()->fill(1.);
    EXPECT_TRUE(sqp.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::NaN_GRADIENT_NORM, sqp.getStoppingCriterion());
    std::shared_ptr<dotk::Vector<Real> > gold = dotk::gtest::allocateData(5, 3.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold);
    gold->fill(1.);
    dotk::gtest::checkResults(*mng->getNewGradient(), *gold);

    // TEST 5: NaN TRIAL STEP NORM
    mng->getOldPrimal()->fill(4);
    mng->getNewGradient()->fill(2);
    mng->getOldGradient()->fill(3);
    mng->getTrialStep()->fill(std::numeric_limits<Real>::quiet_NaN());
    EXPECT_TRUE(sqp.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::NaN_TRIAL_STEP_NORM, sqp.getStoppingCriterion());
    gold->fill(4.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold);
    gold->fill(3.);
    dotk::gtest::checkResults(*mng->getNewGradient(), *gold);

    // TEST 6: ALGORITHM HAS NOT CONVERGED
    mng->getTrialStep()->fill(1);
    EXPECT_FALSE(sqp.checkStoppingCriteria(mng));
    EXPECT_EQ(dotk::types::OPT_ALG_HAS_NOT_CONVERGED, sqp.getStoppingCriterion());
    gold->fill(4.);
    dotk::gtest::checkResults(*mng->getNewPrimal(), *gold);
    gold->fill(3.);
    dotk::gtest::checkResults(*mng->getNewGradient(), *gold);
}

TEST(DOTk_InexactTrustRegionSQPTest, computePartialPredictedReduction)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    setSqpTestData(mng);

    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);

    Real tolerance = 1e-10;
    EXPECT_NEAR(-0.0041573142444964514, sqp.computePartialPredictedReduction(), tolerance);
}

TEST(DOTk_InexactTrustRegionSQPTest, updateMeritFunctionPenaltyParameter)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);

    // TEST 1: PARTIAL REDUCTION GREATER THAN UPDATE CRITERION
    Real tolerance = 1e-10;
    mng->m_LinearizedEqConstraint->fill(0.1);
    mng->getNewEqualityConstraintResidual()->fill(0.1);
    mng->m_LinearizedEqConstraint->update(1., *mng->getNewEqualityConstraintResidual(), 1.);
    Real partial_predicted_reduction = 1.;
    sqp.updateMeritFunctionPenaltyParameter(partial_predicted_reduction);
    EXPECT_NEAR(1., sqp.getMeritFunctionPenaltyParameter(), tolerance);

    // TEST 2: PARTIAL REDUCTION GREATER THAN UPDATE CRITERION
    mng->m_LinearizedEqConstraint->fill(-1.);
    mng->getNewEqualityConstraintResidual()->fill(4.);
    mng->m_LinearizedEqConstraint->update(1., *mng->getNewEqualityConstraintResidual(), 1.);
    partial_predicted_reduction = -11;
    sqp.updateMeritFunctionPenaltyParameter(partial_predicted_reduction);
    EXPECT_NEAR(1.0476190576190476, sqp.getMeritFunctionPenaltyParameter(), tolerance);
}

TEST(DOTk_InexactTrustRegionSQPTest, computePredictedReductionResidual)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);

    mng->m_DeltaDual->fill(1.);
    mng->m_LinearizedEqConstraint->fill(4.);
    mng->m_JacobianTimesTangentialStep->fill(-1.);
    Real predicted_reduction_residual = sqp.computePredictedReductionResidual();

    Real tolerance = 1e-10;
    EXPECT_NEAR(24., predicted_reduction_residual, tolerance);
}

TEST(DOTk_InexactTrustRegionSQPTest, computePredictedReduction)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);

    mng->m_LinearizedEqConstraint->fill(1.);
    mng->getNewEqualityConstraintResidual()->fill(2.);

    // TEST 1: PARTIAL PREDICTED REDUCTION = 0.
    Real tolerance = 1e-10;
    Real partial_predicted_reduction = 0.;
    Real predicted_reduction = sqp.computePredictedReduction(partial_predicted_reduction);
    EXPECT_NEAR(9., predicted_reduction, tolerance);

    // TEST 2: PARTIAL PREDICTED REDUCTION = -2.
    partial_predicted_reduction = -2.;
    predicted_reduction = sqp.computePredictedReduction(partial_predicted_reduction);
    EXPECT_NEAR(7., predicted_reduction, tolerance);
}

TEST(DOTk_InexactTrustRegionSQPTest, computeActualReduction)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();

    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    solver_mng->setDefaultKrylovSolvers(primal, hessian);

    mng->getOldDual()->fill(0.1);
    mng->getNewDual()->fill(0.1);
    mng->getOldPrimal()->fill(1);
    Real objective_function_value = mng->getRoutinesMng()->objective(mng->getOldPrimal());
    mng->setOldObjectiveFunctionValue(objective_function_value);
    mng->getRoutinesMng()->equalityConstraint(mng->getOldPrimal(), mng->getOldEqualityConstraintResidual());
    mng->getNewPrimal()->fill(1.2);

    // TEST 1: ACTUAL REDUCTION UPDATED.
    Real tolerance = 1e-10;
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);
    Real actual_reduction = sqp.computeActualReduction();
    EXPECT_NEAR(-14.9579163594485, actual_reduction, tolerance);

    // TEST 2: ACTUAL REDUCTION = 0.
    mng->getNewPrimal()->fill(1);
    actual_reduction = sqp.computeActualReduction();
    EXPECT_NEAR(std::numeric_limits<Real>::epsilon(), actual_reduction, tolerance);
}

TEST(DOTk_InexactTrustRegionSQPTest, updateTrustRegionRadius)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    hessian->setFullSpaceHessian();
    solver_mng->setDefaultKrylovSolvers(primal, hessian);

    mng->getTrialStep()->fill(0.1);
    mng->getNormalStep()->fill(0.2);
    mng->getTangentialStep()->fill(0.1);

    // TEST 1
    Real tolerance = 1e-10;
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);
    Real actual_over_predicted_reduction = 0.1;
    EXPECT_FALSE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(0.22360679775, mng->getTrustRegionRadius(), tolerance);

    // TEST 2
    mng->getTangentialStep()->fill(0.3);
    EXPECT_FALSE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(0.33541019662, mng->getTrustRegionRadius(), tolerance);

    // TEST 3
    actual_over_predicted_reduction = 0.91;
    EXPECT_TRUE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(1.5652475842, mng->getTrustRegionRadius(), tolerance);

    // TEST 4
    mng->setMaxTrustRegionRadius(1.);
    EXPECT_TRUE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(1., mng->getTrustRegionRadius(), tolerance);

    // TEST 5
    mng->setMaxTrustRegionRadius(1e4);
    mng->setMinTrustRegionRadius(1e2);
    EXPECT_TRUE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(1e2, mng->getTrustRegionRadius(), tolerance);

    // TEST 6
    actual_over_predicted_reduction = 0.9;
    mng->setMinTrustRegionRadius(1e-8);
    EXPECT_TRUE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(1e2, mng->getTrustRegionRadius(), tolerance);

    // TEST 7
    mng->setTrustRegionRadius(0.1);
    EXPECT_TRUE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(0.44721359550, mng->getTrustRegionRadius(), tolerance);

    // TEST 8
    mng->setMaxTrustRegionRadius(0.3);
    EXPECT_TRUE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(0.3, mng->getTrustRegionRadius(), tolerance);

    // TEST 9
    mng->setMaxTrustRegionRadius(1e4);
    mng->setMinTrustRegionRadius(1e2);
    EXPECT_TRUE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(1e2, mng->getTrustRegionRadius(), tolerance);

    // TEST 10
    actual_over_predicted_reduction = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_FALSE(sqp.updateTrustRegionRadius(actual_over_predicted_reduction));
    EXPECT_NEAR(50, mng->getTrustRegionRadius(), tolerance);
}

TEST(DOTk_InexactTrustRegionSQPTest, adjustSolversTolerance)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();

    // TEST 1: TOLERANCES ADJUSTED
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        sqp_solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    sqp_solver_mng->setDefaultKrylovSolvers(primal, hessian);

    EXPECT_FALSE(sqp_solver_mng->adjustSolversTolerance());
    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-5, sqp_solver_mng->getDualTolerance(), tolerance);
    EXPECT_NEAR(1e-5, sqp_solver_mng->getTangentialTolerance(), tolerance);
    EXPECT_NEAR(1e-5, sqp_solver_mng->getQuasiNormalProblemRelativeTolerance(), tolerance);
    EXPECT_NEAR(1e-5, sqp_solver_mng->getTangentialSubProbLeftPrecProjectionTolerance(), tolerance);

    // TEST 2: TOLERANCES NOT ADJUSTED, LOWER BOUND VIOLATED
    sqp_solver_mng->setQuasiNormalProblemRelativeTolerance(1e-18);
    EXPECT_TRUE(sqp_solver_mng->adjustSolversTolerance());
    EXPECT_NEAR(1e-5, sqp_solver_mng->getDualTolerance(), tolerance);
    EXPECT_NEAR(1e-5, sqp_solver_mng->getTangentialTolerance(), tolerance);
    EXPECT_NEAR(1e-18, sqp_solver_mng->getQuasiNormalProblemRelativeTolerance(), tolerance);
    EXPECT_NEAR(1e-5, sqp_solver_mng->getTangentialSubProbLeftPrecProjectionTolerance(), tolerance);
}

TEST(DOTk_InexactTrustRegionSQPTest, getMin)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals, 1.);
    primal->allocateSerialControlArray(ncontrols);
    setSqpTestInitialGuess(primal);

    std::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));
    std::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng>
        solver_mng(new dotk::DOTk_InexactTrustRegionSqpSolverMng(primal, mng));
    std::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);

    hessian->setFullSpaceHessian();
    mng->setMaxTrustRegionRadius(1e2);
    solver_mng->setDefaultKrylovSolvers(primal, hessian);
    dotk::DOTk_InexactTrustRegionSQP sqp(hessian, mng, solver_mng);
    sqp.getMin();

    Real tolerance = 1e-8;
    EXPECT_NEAR(-1.717143570394393, (*mng->getNewPrimal())[0], tolerance);
    EXPECT_NEAR(1.595709690183567, (*mng->getNewPrimal())[1], tolerance);
    EXPECT_NEAR(1.827245752927180, (*mng->getNewPrimal())[2], tolerance);
    EXPECT_NEAR(-0.763643078184132, (*mng->getNewPrimal())[3], tolerance);
    EXPECT_NEAR(-0.763643078184132, (*mng->getNewPrimal())[4], tolerance);
    EXPECT_EQ(19u, sqp.getNumItrDone());
}

void setSqpTestInitialGuess(std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    assert(primal_->control().use_count() > 0);
    (*primal_->control())[0] = -1.8;
    (*primal_->control())[1] = 1.7;
    (*primal_->control())[2] = 1.9;
    (*primal_->control())[3] = -0.8;
    (*primal_->control())[4] = -0.8;
}

void setSqpTestData(const std::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_)
{
    (*mng_->getOldPrimal())[0] = -1.8;
    (*mng_->getOldPrimal())[1] = 1.7;
    (*mng_->getOldPrimal())[2] = 1.9;
    (*mng_->getOldPrimal())[3] = -0.8;
    (*mng_->getOldPrimal())[4] = -0.8;
    (*mng_->getProjectedGradient())[0] = -0.00166757862663009;
    (*mng_->getProjectedGradient())[1] = 0.00186953451566834;
    (*mng_->getProjectedGradient())[2] = -0.00307601487523406;
    (*mng_->getProjectedGradient())[3] = 0.000209638713516019;
    (*mng_->getProjectedGradient())[4] = 0.000209638713515999;
    (*mng_->getProjectedTangentialStep())[0] = 0.0196000125684099;
    (*mng_->getProjectedTangentialStep())[1] = -0.021973716512681;
    (*mng_->getProjectedTangentialStep())[2] = 0.0361541754328189;
    (*mng_->getProjectedTangentialStep())[3] = -0.0024640046077124;
    (*mng_->getProjectedTangentialStep())[4] = -0.00246400460771216;
    (*mng_->getOldGradient())[0] = -0.00110961976603019;
    (*mng_->getOldGradient())[1] = 0.00124400278267656;
    (*mng_->getOldGradient())[2] = -0.00204680418161685;
    (*mng_->getOldGradient())[3] = 0.000139495227707814;
    (*mng_->getOldGradient())[4] = 0.000139495227707814;
    (*mng_->getNewGradient())[0] = 0.592218134739563;
    (*mng_->getNewGradient())[1] = 0.42604315139002;
    (*mng_->getNewGradient())[2] = -0.0387374769570285;
    (*mng_->getNewGradient())[3] = 0.0952340715449905;
    (*mng_->getNewGradient())[4] = 0.0952340715449906;
    (*mng_->getNormalStep())[0] = 0.0720254505984308;
    (*mng_->getNormalStep())[1] = -0.09009081658786;
    (*mng_->getNormalStep())[2] = -0.0888237903661056;
    (*mng_->getNormalStep())[3] = 0.0365216243924142;
    (*mng_->getNormalStep())[4] = 0.0365216243924142;
    (*mng_->getHessTimesNormalStep())[0] = 0.821886987715101;
    (*mng_->getHessTimesNormalStep())[1] = 0.661914924684899;
    (*mng_->getHessTimesNormalStep())[2] = -0.0339371478280196;
    (*mng_->getHessTimesNormalStep())[3] = 0.0810737062636601;
    (*mng_->getHessTimesNormalStep())[4] = 0.0810737062636601;
    (*mng_->getMatrixTimesVector())[0] = 0.000656769787120568;
    (*mng_->getMatrixTimesVector())[1] = -0.00266329653273746;
    (*mng_->getMatrixTimesVector())[2] = 0.00301826782503528;
    (*mng_->getMatrixTimesVector())[3] = -0.00111421522824808;
    (*mng_->getMatrixTimesVector())[4] = -0.00111421522824806;
    (*mng_->m_DeltaDual)[0] = 0.0187231394430601;
    (*mng_->m_DeltaDual)[1] = -0.0174179992675516;
    (*mng_->m_DeltaDual)[2] = -0.060159190502251;
    (*mng_->getOldDual())[0] = 0.0207961931305597;
    (*mng_->getOldDual())[1] = -0.019798867645168;
    (*mng_->getOldDual())[2] = 0.0834391241457092;
    (*mng_->getNewDual())[0] = 0.0395193325736199;
    (*mng_->getNewDual())[1] = -0.0372168669127196;
    (*mng_->getNewDual())[2] = 0.0232799336434582;
    (*mng_->m_LinearizedEqConstraint)[0] = 0;
    (*mng_->m_LinearizedEqConstraint)[1] = 0;
    (*mng_->m_LinearizedEqConstraint)[2] = -1.11022302462516e-16;
    (*mng_->m_JacobianTimesTangentialStep)[0] = 5.46437894932694e-17;
    (*mng_->m_JacobianTimesTangentialStep)[1] = -8.67361737988404e-17;
    (*mng_->m_JacobianTimesTangentialStep)[2] = -5.55111512312578e-16;
}


}
