/*
 * DOTk_DoglegTrustRegionTest.cpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_DoglegTrustRegion.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkDoglegTrustRegionTest
{

TEST(DOTk_DoglegTrustRegion, dogleg)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > gradient = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > vector = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > newton_dir = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > conjugate_dir = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > matrix_times_grad = primal->control()->clone();

    vector->fill(2);
    mng.getRoutinesMng()->gradient(vector, gradient);
    mng.getRoutinesMng()->hessian(vector, gradient, matrix_times_grad);
    newton_dir->update(1., *gradient, 0.);

    dotk::DOTk_DoglegTrustRegion step;
    EXPECT_EQ(dotk::types::TRUST_REGION_DOGLEG, step.getTrustRegionType());
    step.dogleg(gradient, matrix_times_grad, conjugate_dir, newton_dir);

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -3855.6568684884;
    (*gold)[1] = 962.71082858637;
    dotk::gtest::checkResults(*newton_dir, *gold);
}

TEST(DOTk_DoglegTrustRegion, step)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > vector = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > conjugate_dir = primal->control()->clone();

    vector->fill(2);
    mng.getRoutinesMng()->gradient(vector, mng.getNewGradient());
    mng.getRoutinesMng()->hessian(vector, mng.getNewGradient(), mng.getMatrixTimesVector());
    mng.getTrialStep()->update(1., *mng.getNewGradient(), 0.);

    dotk::DOTk_DoglegTrustRegion step;
    EXPECT_EQ(dotk::types::TRUST_REGION_DOGLEG, step.getTrustRegionType());
    step.step(&mng, conjugate_dir, mng.getTrialStep());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -3855.6568684884;
    (*gold)[1] = 962.71082858637;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
}

}
