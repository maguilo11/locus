/*
 * DOTk_DoubleDoglegTrustRegionTest.cpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_DoubleDoglegTrustRegion.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkDoubleDoglegTrustRegionTest
{

TEST(DOTk_DoubleDoglegTrustRegion, setAndGetParamPromotesMonotonicallyDecreasingQuadraticModel)
{
    std::tr1::shared_ptr<dotk::vector<Real> > vec = dotk::gtest::allocateData(2, 0);
    dotk::DOTk_DoubleDoglegTrustRegion step(vec);
    EXPECT_EQ(dotk::types::TRUST_REGION_DOUBLE_DOGLEG, step.getTrustRegionType());

    Real tol = 1e-8;
    EXPECT_NEAR(0.8, step.getParamPromotesMonotonicallyDecreasingQuadraticModel(), tol);
    step.setParamPromotesMonotonicallyDecreasingQuadraticModel(0.4);
    EXPECT_NEAR(0.4, step.getParamPromotesMonotonicallyDecreasingQuadraticModel(), tol);
}

TEST(DOTk_DoubleDoglegTrustRegion, computeDoubleDoglegRoot)
{
    std::tr1::shared_ptr<dotk::vector<Real> > grad = dotk::gtest::allocateData(2, 0);
    (*grad)[0] = 6.;
    (*grad)[1] = 2.;

    std::tr1::shared_ptr<dotk::vector<Real> > hess_times_grad = grad->clone();
    (*hess_times_grad)[0] = 84.;
    (*hess_times_grad)[1] = 4.;

    std::tr1::shared_ptr<dotk::vector<Real> > newton_direction = grad->clone();
    (*newton_direction)[0] = -3. / 7.;
    (*newton_direction)[1] = -1.;

    dotk::DOTk_DoubleDoglegTrustRegion step(grad);
    Real root = step.computeDoubleDoglegRoot(grad, newton_direction, hess_times_grad);

    Real tolerance = 1e-6;
    EXPECT_NEAR(0.746875, root, tolerance);
}

TEST(DOTk_DoubleDoglegTrustRegion, doubleDogleg)
{
    std::tr1::shared_ptr<dotk::vector<Real> > grad = dotk::gtest::allocateData(2, 0);
    (*grad)[0] = 6.;
    (*grad)[1] = 2.;

    std::tr1::shared_ptr<dotk::vector<Real> > hess_times_grad = grad->clone();
    (*hess_times_grad)[0] = 84.;
    (*hess_times_grad)[1] = 4.;

    std::tr1::shared_ptr<dotk::vector<Real> > newton_dir = grad->clone();
    (*newton_dir)[0] = -3. / 7.;
    (*newton_dir)[1] = -1.;

    Real trust_region_radius = 0.75;
    dotk::DOTk_DoubleDoglegTrustRegion step(grad);
    step.doubleDogleg(trust_region_radius, grad, hess_times_grad, newton_dir);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = grad->clone();
    (*gold)[0] = -0.3397877009;
    (*gold)[1] = -0.6686137287;
    dotk::gtest::checkResults(*newton_dir, *gold);
}

TEST(DOTk_DoubleDoglegTrustRegion, step)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    (*primal->control())[0] = 6.;
    (*primal->control())[1] = 2.;
    mng.setNewGradient(*primal->control());

    (*primal->control())[0] = -3. / 7.;
    (*primal->control())[1] = -1.;
    mng.setTrialStep(*primal->control());

    (*mng.getMatrixTimesVector())[0] = 84.;
    (*mng.getMatrixTimesVector())[1] = 4.;
    dotk::DOTk_DoubleDoglegTrustRegion step(mng.getTrialStep());
    step.setTrustRegionRadius(0.75);
    step.step(&mng, mng.getMatrixTimesVector(), mng.getTrialStep());

    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -0.3397877009;
    (*gold)[1] = -0.6686137287;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
}

}
