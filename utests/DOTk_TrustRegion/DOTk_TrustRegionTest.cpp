/*
 * DOTk_TrustRegionTest.cpp
 *
 *  Created on: Sep 9, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_FirstOrderOperator.hpp"
#include "DOTk_LineSearchAlgorithmsDataMng.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_LineSearch.hpp"
#include "DOTk_TrustRegion.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkTrustRegionTest
{

TEST(DOTk_TrustRegion, setAndGetLineSearchIterationCount)
{
    dotk::DOTk_TrustRegion step;
    EXPECT_EQ(-1, step.getLineSearchIterationCount());
    step.setLineSearchIterationCount(3);
    EXPECT_EQ(3, step.getLineSearchIterationCount());
}

TEST(DOTk_TrustRegion, setAndGetMaxTrustRegionSubProblemIterations)
{
    dotk::DOTk_TrustRegion step;
    EXPECT_EQ(50, step.getMaxTrustRegionSubProblemIterations());
    step.setMaxTrustRegionSubProblemIterations(31);
    EXPECT_EQ(31, step.getMaxTrustRegionSubProblemIterations());
}

TEST(DOTk_TrustRegion, setAndGetNumTrustRegionSubProblemItrDone)
{
    dotk::DOTk_TrustRegion step;
    EXPECT_EQ(0, step.getNumTrustRegionSubProblemItrDone());
    step.setNumTrustRegionSubProblemItrDone(21);
    EXPECT_EQ(21, step.getNumTrustRegionSubProblemItrDone());
}

TEST(DOTk_TrustRegion, invalidCurvatureDetected)
{
    dotk::DOTk_TrustRegion step;

    EXPECT_FALSE(step.isCurvatureInvalid());
    step.invalidCurvatureDetected(true);
    EXPECT_TRUE(step.isCurvatureInvalid());
}

TEST(DOTk_TrustRegion, setAndGetLineSearchStep)
{
    dotk::DOTk_TrustRegion step;
    Real tol = 1e-8;
    EXPECT_NEAR(1, step.getLineSearchStep(), tol);
    step.setLineSearchStep(0.12);
    EXPECT_NEAR(0.12, step.getLineSearchStep(), tol);
}

TEST(DOTk_TrustRegion, setAndGetTrustRegionRadius)
{
    dotk::DOTk_TrustRegion step;
    Real tol = 1e-8;
    EXPECT_NEAR(1e4, step.getTrustRegionRadius(), tol);
    step.setTrustRegionRadius(1e1);
    EXPECT_NEAR(1e1, step.getTrustRegionRadius(), tol);
}

TEST(DOTk_TrustRegion, setAndGetMaxTrustRegionRadius)
{
    dotk::DOTk_TrustRegion step;
    Real tol = 1e-8;
    EXPECT_NEAR(1e4, step.getMaxTrustRegionRadius(), tol);
    step.setMaxTrustRegionRadius(1e2);
    EXPECT_NEAR(1e2, step.getMaxTrustRegionRadius(), tol);
}

TEST(DOTk_TrustRegion, setAndGetMinTrustRegionRadius)
{
    dotk::DOTk_TrustRegion step;
    Real tol = 1e-8;
    EXPECT_NEAR(1e-6, step.getMinTrustRegionRadius(), tol);
    step.setMinTrustRegionRadius(1e-2);
    EXPECT_NEAR(1e-2, step.getMinTrustRegionRadius(), tol);
}

TEST(DOTk_TrustRegion, setAndGetActualReduction)
{
    dotk::DOTk_TrustRegion step;

    Real tol = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::min(), step.getActualReduction(), tol);
    Real old_objective_func_val = 1.75;
    Real new_objective_func_val = 1.;
    step.computeActualReduction(new_objective_func_val, old_objective_func_val);
    EXPECT_NEAR(0.75, step.getActualReduction(), tol);
}

TEST(DOTk_TrustRegion, setAndGetPredictedReduction)
{
    dotk::DOTk_TrustRegion step;

    Real tol = 1e-8;
    EXPECT_NEAR(std::numeric_limits<Real>::min(), step.getPredictedReduction(), tol);

    std::tr1::shared_ptr<dotk::vector<Real> > grad = dotk::gtest::allocateData(2, 2);
    std::tr1::shared_ptr<dotk::vector<Real> > trial_step = grad->clone();
    trial_step->fill(2);
    std::tr1::shared_ptr<dotk::vector<Real> > hess_times_trial_step = grad->clone();
    hess_times_trial_step->fill(2);
    step.computePredictedReduction(grad, trial_step, hess_times_trial_step);
    EXPECT_NEAR(-12., step.getPredictedReduction(), tol);
}

TEST(DOTk_TrustRegion, setAndGetContractionParameter)
{
    dotk::DOTk_TrustRegion step;
    Real tol = 1e-8;
    EXPECT_NEAR(0.5, step.getContractionParameter(), tol);
    step.setContractionParameter(0.25);
    EXPECT_NEAR(0.25, step.getContractionParameter(), tol);
}

TEST(DOTk_TrustRegion, setAndGetExpansionParameter)
{
    dotk::DOTk_TrustRegion step;
    Real tol = 1e-8;
    EXPECT_NEAR(2., step.getExpansionParameter(), tol);
    step.setExpansionParameter(3.);
    EXPECT_NEAR(3., step.getExpansionParameter(), tol);
}

TEST(DOTk_TrustRegion, setAndGetMinActualOverPredictedReductionAllowed)
{
    dotk::DOTk_TrustRegion step;
    Real tol = 1e-8;
    EXPECT_NEAR(0.25, step.getMinActualOverPredictedReductionAllowed(), tol);
    step.setMinActualOverPredictedReductionAllowed(0.1);
    EXPECT_NEAR(0.1, step.getMinActualOverPredictedReductionAllowed(), tol);
}

TEST(DOTk_TrustRegion, setAndGetTrustRegionType)
{
    dotk::DOTk_TrustRegion step;
    EXPECT_EQ(dotk::types::TRUST_REGION_DISABLED, step.getTrustRegionType());
    step.setTrustRegionType(dotk::types::TRUST_REGION_DOGLEG);
    EXPECT_EQ(dotk::types::TRUST_REGION_DOGLEG, step.getTrustRegionType());
}

TEST(DOTk_TrustRegion, isTrustRegionStepInvalid)
{
    dotk::DOTk_TrustRegion step;
    Real trust_region_step = 1.;
    EXPECT_FALSE(step.isTrustRegionStepInvalid(trust_region_step));
    trust_region_step = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_TRUE(step.isTrustRegionStepInvalid(trust_region_step));
    trust_region_step = std::numeric_limits<Real>::infinity();
    EXPECT_TRUE(step.isTrustRegionStepInvalid(trust_region_step));
}

TEST(DOTk_TrustRegion, computeAlternateStep)
{
    dotk::DOTk_TrustRegion step;

    // TEST: norm of Newton direction is greater that trust region radius
    Real tol = 1e-8;
    std::tr1::shared_ptr<dotk::vector<Real> > vec = dotk::gtest::allocateData(2);
    vec->fill(1e4);
    Real trust_region_step = step.computeAlternateStep(step.getTrustRegionRadius(), vec);
    EXPECT_NEAR(1e4, trust_region_step, tol);
    // TEST: norm of Newton direction is less that trust region radius
    vec->fill(2.);
    trust_region_step = step.computeAlternateStep(step.getTrustRegionRadius(), vec);
    EXPECT_NEAR(1., trust_region_step, tol);
}

TEST(DOTk_TrustRegion, computeDoglegStep)
{
    dotk::DOTk_TrustRegion step;

    std::tr1::shared_ptr<dotk::vector<Real> > cauchy_dir = dotk::gtest::allocateData(2);
    (*cauchy_dir)[0] = -0.46875;
    (*cauchy_dir)[1] = -0.15625;
    std::tr1::shared_ptr<dotk::vector<Real> > newton_dir = cauchy_dir->clone();
    (*newton_dir)[0] = -0.320;
    (*newton_dir)[1] = -0.747;
    Real trust_region_radius = 0.75;
    newton_dir->axpy(-1, *cauchy_dir);
    Real root = step.computeDoglegRoot(trust_region_radius, cauchy_dir, newton_dir);
    const Real tol = 1e-8;
    EXPECT_NEAR(0.797731933418, root, tol);
}

TEST(DOTk_TrustRegion, acceptTrustRegionRadius)
{
    // Test: NaN actual over predicted reduction
    std::tr1::shared_ptr<dotk::vector<Real> > trial_step = dotk::gtest::allocateData(2);

    dotk::DOTk_TrustRegion step;
    step.setMaxTrustRegionRadius(1e10);
    step.setTrustRegionRadius(1.);

    EXPECT_FALSE(step.acceptTrustRegionRadius(trial_step));
    const Real tolerance = 1e-8;
    EXPECT_NEAR(0.5, step.getTrustRegionRadius(), tolerance);

    // Test: actual over predicted reduction is less than lower limit
    std::tr1::shared_ptr<dotk::vector<Real> > grad = trial_step->clone();
    grad->fill(std::numeric_limits<Real>::infinity());
    std::tr1::shared_ptr<dotk::vector<Real> > hess_times_vec = trial_step->clone();
    hess_times_vec->fill(std::numeric_limits<Real>::infinity());

    step.setTrustRegionRadius(1.);
    step.computeActualReduction(1., 2.);
    step.computePredictedReduction(grad, trial_step, hess_times_vec);

    EXPECT_FALSE(step.acceptTrustRegionRadius(trial_step));
    EXPECT_NEAR(0.5, step.getTrustRegionRadius(), tolerance);

    // Test: actual an predicted reduction is inside allowable range, keep current trust region
    grad->fill(1);
    trial_step->fill(-1);
    hess_times_vec->fill(1);

    step.setTrustRegionRadius(1.);
    step.computeActualReduction(1., 2.);
    step.computePredictedReduction(grad, trial_step, hess_times_vec);

    EXPECT_TRUE(step.acceptTrustRegionRadius(trial_step));
    EXPECT_NEAR(1., step.getTrustRegionRadius(), tolerance);

    // Test: actual an predicted reduction is inside allowable range and close to trust region boundary, expand trust region by 2
    step.setTrustRegionRadius(1.4142135623731);
    EXPECT_TRUE(step.acceptTrustRegionRadius(trial_step));
    EXPECT_NEAR(2.8284271247462, step.getTrustRegionRadius(), tolerance);
}

TEST(DOTk_TrustRegion, computeCauchyDirection)
{
    dotk::DOTk_TrustRegion step;
    step.setTrustRegionRadius(1.);

    std::tr1::shared_ptr<dotk::vector<Real> > grad = dotk::gtest::allocateData(2);
    (*grad)[0] = 6.;
    (*grad)[1] = 2.;

    std::tr1::shared_ptr<dotk::vector<Real> > matrix_times_grad = grad->clone();
    (*matrix_times_grad)[0] = 84;
    (*matrix_times_grad)[1] = 4;

    std::tr1::shared_ptr<dotk::vector<Real> > cauchy_point = grad->clone();
    step.computeCauchyPoint(grad, matrix_times_grad, cauchy_point);

    // TEST: Positive curvature
    std::tr1::shared_ptr<dotk::vector<Real> > gold = grad->clone();
    (*gold)[0] = -0.46875;
    (*gold)[1] = -0.15625;
    dotk::gtest::checkResults(*cauchy_point, *gold);

    // TEST: Negative curvature
    matrix_times_grad->scale(-1);
    step.computeCauchyPoint(grad, matrix_times_grad, cauchy_point);
    (*gold)[0] = -0.9486832980505;
    (*gold)[1] = -0.3162277660168;
    dotk::gtest::checkResults(*cauchy_point, *gold);
}

}
