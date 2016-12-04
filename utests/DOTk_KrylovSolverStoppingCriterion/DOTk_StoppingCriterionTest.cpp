/*
 * DOTk_StoppingCriterionTest.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_NocedalAndWrightEquality.hpp"
#include "DOTk_NocedalAndWrightObjective.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Variable.hpp"
#include "DOTk_PrecGMRES.hpp"
#include "DOTk_KrylovSolver.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_LinearOperator.hpp"
#include "DOTk_AugmentedSystem.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_KrylovSolverDataMng.hpp"
#include "DOTk_PrecGenMinResDataMng.hpp"
#include "DOTk_TrustRegionAlgorithmsDataMng.hpp"
#include "DOTk_KrylovSolverStoppingCriterion.hpp"
#include "DOTk_FixedCriterion.hpp"
#include "DOTk_RelativeCriterion.hpp"
#include "DOTk_QuasiNormalProbCriterion.hpp"
#include "DOTk_TangentialProblemCriterion.hpp"
#include "DOTk_TangentialSubProblemCriterion.hpp"


namespace DOTkStoppingCriterionTest
{

TEST(DOTk_KrylovSolverStoppingCriterion, DOTk_FixedCriterion)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols, 1.);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    std::tr1::shared_ptr<dotk::DOTk_AugmentedSystem> augmented_system(new dotk::DOTk_AugmentedSystem);
    std::tr1::shared_ptr<dotk::DOTk_PrecGenMinResDataMng> smng(new dotk::DOTk_PrecGenMinResDataMng(primal, augmented_system));
    std::tr1::shared_ptr<dotk::DOTk_PrecGMRES> solver(new dotk::DOTk_PrecGMRES(smng));

    Real fixed_tolerance = 1e-4;
    dotk::DOTk_FixedCriterion tol(fixed_tolerance);

    Real tolerance = 1e-8;
    std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> >
        vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    EXPECT_NEAR(1e-4, tol.evaluate(solver.get(), vector), tolerance);
}

TEST(DOTk_KrylovSolverStoppingCriterion, DOTk_RelativeCriterion)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols, 1.);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    std::tr1::shared_ptr<dotk::DOTk_AugmentedSystem> augmented_system(new dotk::DOTk_AugmentedSystem);
    std::tr1::shared_ptr<dotk::DOTk_PrecGenMinResDataMng> smng(new dotk::DOTk_PrecGenMinResDataMng(primal, augmented_system));
    std::tr1::shared_ptr<dotk::DOTk_PrecGMRES> solver(new dotk::DOTk_PrecGMRES(smng));

    Real relative_fixed_tolerance = 1e-4;
    dotk::DOTk_RelativeCriterion tol(relative_fixed_tolerance);

    // TEST 1: INITIAL CALCULATION, E.G. SOLVER ITERATION = 0
    Real tolerance = 1e-8;
    std::tr1::shared_ptr<dotk::vector<Real> >
        vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    solver->setSolverResidualNorm(0.2);
    EXPECT_NEAR(2e-5, tol.evaluate(solver.get(), vector), tolerance);

    // TEST 2:SOLVER ITERATION > 0 AND THUS TOLERANCE WAS ALREADY SET. HENCE, A NEW RESIDUAL NORM SHOULD NOT CHANGE THE VALUES
    solver->setNumSolverItrDone(1);
    solver->setSolverResidualNorm(0.8);
    EXPECT_NEAR(2e-5, tol.evaluate(solver.get(), vector), tolerance);
}

TEST(DOTk_KrylovSolverStoppingCriterion, DOTk_TangentialProblemCriterion)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols, 1.);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    std::tr1::shared_ptr<dotk::DOTk_AugmentedSystem> augmented_system(new dotk::DOTk_AugmentedSystem);
    std::tr1::shared_ptr<dotk::DOTk_PrecGenMinResDataMng> smng(new dotk::DOTk_PrecGenMinResDataMng(primal, augmented_system));
    std::tr1::shared_ptr<dotk::DOTk_PrecGMRES> solver(new dotk::DOTk_PrecGMRES(smng));

    dotk::DOTk_TangentialProblemCriterion tol(mng->getTrialStep());

    // TEST 1: TOLERANCE < EPSILON
    std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> >
        vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    EXPECT_EQ(std::numeric_limits<Real>::epsilon(), tol.evaluate(solver.get(), vector));

    // TEST 2: TOLERANCE = MINIMUM ELEMENT
    mng->getTrialStep()->fill(1);
    tol.setCurrentTrialStep(mng->getTrialStep());
    tol.set(dotk::types::TRUST_REGION_RADIUS, 1);
    Real norm_projected_tangential_step = 0.1;
    tol.set(dotk::types::NORM_PROJECTED_TANGENTIAL_STEP, norm_projected_tangential_step);
    Real norm_tangential_problem_residual = 0.2;
    tol.set(dotk::types::NORM_TANGENTIAL_STEP_RESIDUAL, norm_tangential_problem_residual);

    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-5, tol.evaluate(solver.get(), vector), tolerance);
}

TEST(DOTk_KrylovSolverStoppingCriterion, DOTk_QuasiNormalProbCriterion)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols, 1.);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    std::tr1::shared_ptr<dotk::DOTk_AugmentedSystem> augmented_system(new dotk::DOTk_AugmentedSystem);
    std::tr1::shared_ptr<dotk::DOTk_PrecGenMinResDataMng> smng(new dotk::DOTk_PrecGenMinResDataMng(primal, augmented_system));
    std::tr1::shared_ptr<dotk::DOTk_PrecGMRES> solver(new dotk::DOTk_PrecGMRES(smng));

    dotk::DOTk_QuasiNormalProbCriterion tol;

    Real tolerance = 1e-8;
    EXPECT_NEAR(1e-8, tol.getStoppingTolerance(), tolerance);
    tol.setStoppingTolerance(1);
    EXPECT_NEAR(1, tol.getStoppingTolerance(), tolerance);

    EXPECT_NEAR(1e-4, tol.getRelativeTolerance(), tolerance);
    tol.setRelativeTolerance(2);
    EXPECT_NEAR(2, tol.getRelativeTolerance(), tolerance);

    EXPECT_NEAR(0.8, tol.getTrustRegionRadiusPenaltyParameter(), tolerance);
    tol.setTrustRegionRadiusPenaltyParameter(0.7);
    EXPECT_NEAR(0.7, tol.getTrustRegionRadiusPenaltyParameter(), tolerance);

    tol.setStoppingTolerance(1e-3);
    std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> >
        vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    EXPECT_NEAR(1e-3, tol.evaluate(solver.get(), vector), tolerance);
}

TEST(DOTk_KrylovSolverStoppingCriterion, DOTk_TangentialSubProblemCriterion)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols, 1.);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    std::tr1::shared_ptr<dotk::DOTk_AugmentedSystem> augmented_system(new dotk::DOTk_AugmentedSystem);
    std::tr1::shared_ptr<dotk::DOTk_PrecGenMinResDataMng> smng(new dotk::DOTk_PrecGenMinResDataMng(primal, augmented_system));
    std::tr1::shared_ptr<dotk::DOTk_PrecGMRES> solver(new dotk::DOTk_PrecGMRES(smng));

    Real projected_gradient_tolerance = 1e-4;
    dotk::DOTk_TangentialSubProblemCriterion tol(projected_gradient_tolerance);

    // TEST 1: INITIAL CALCULATION, E.G. SOLVER ITERATION = 0, INVALID TOLERANCE
    tol.set(dotk::types::TRUST_REGION_RADIUS, 1.);
    tol.set(dotk::types::CURRENT_KRYLOV_SOLVER_ITR, 0);

    mng->getNewGradient()->fill(1);
    Real gradient_dot_gradient = mng->getNewGradient()->dot(*mng->getNewGradient());
    Real norm_gradient = std::sqrt(gradient_dot_gradient);
    tol.set(dotk::types::NORM_GRADIENT, norm_gradient);

    std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> >
        vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    vector->fill(0);
    EXPECT_EQ(std::numeric_limits<Real>::epsilon(), tol.evaluate(solver.get(), vector));

    // TEST 2: INITIAL CALCULATION, E.G. SOLVER ITERATION = 0, VALID TOLERANCE
    Real tolerance = 1e-8;
    tol.set(dotk::types::TRUST_REGION_RADIUS, 1.);
    vector->fill(1);
    EXPECT_NEAR(1e-4, tol.evaluate(solver.get(), vector), tolerance);

    // TEST 3: INITIAL CALCULATION, E.G. SOLVER ITERATION > 0, VALID TOLERANCE
    tol.set(dotk::types::NORM_RESIDUAL, 0.8);
    tol.set(dotk::types::CURRENT_KRYLOV_SOLVER_ITR, 1.);
    EXPECT_NEAR(8e-5, tol.evaluate(solver.get(), vector), tolerance);

    // TEST 4: INITIAL CALCULATION, E.G. SOLVER ITERATION > 0, INVALID TOLERANCE
    vector->fill(0);
    EXPECT_EQ(std::numeric_limits<Real>::epsilon(), tol.evaluate(solver.get(), vector));
}

}
