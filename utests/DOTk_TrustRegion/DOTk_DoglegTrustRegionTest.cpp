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
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    std::tr1::shared_ptr<dotk::vector<Real> > grad = primal->control()->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > vector = primal->control()->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > newton_dir = primal->control()->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > conjugate_dir = primal->control()->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > matrix_times_grad = primal->control()->clone();

    vector->fill(2);
    mng.getRoutinesMng()->gradient(vector, grad);
    mng.getRoutinesMng()->hessian(vector, grad, matrix_times_grad);
    newton_dir->copy(*grad);

    dotk::DOTk_DoglegTrustRegion step;
    EXPECT_EQ(dotk::types::TRUST_REGION_DOGLEG, step.getTrustRegionType());
    step.dogleg(grad, matrix_times_grad, conjugate_dir, newton_dir);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -3855.6568684884;
    (*gold)[1] = 962.71082858637;
    dotk::gtest::checkResults(*newton_dir, *gold);
}

TEST(DOTk_DoglegTrustRegion, step)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    std::tr1::shared_ptr<dotk::vector<Real> > vector = primal->control()->clone();
    std::tr1::shared_ptr<dotk::vector<Real> > conjugate_dir = primal->control()->clone();

    vector->fill(2);
    mng.getRoutinesMng()->gradient(vector, mng.getNewGradient());
    mng.getRoutinesMng()->hessian(vector, mng.getNewGradient(), mng.getMatrixTimesVector());
    mng.getTrialStep()->copy(*mng.getNewGradient());

    dotk::DOTk_DoglegTrustRegion step;
    EXPECT_EQ(dotk::types::TRUST_REGION_DOGLEG, step.getTrustRegionType());
    step.step(&mng, conjugate_dir, mng.getTrialStep());

    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = -3855.6568684884;
    (*gold)[1] = 962.71082858637;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
}

}
