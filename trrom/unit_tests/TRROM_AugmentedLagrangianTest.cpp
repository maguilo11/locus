/*
 * TRROM_AugmentedLagrangianTest.cpp
 *
 *  Created on: Aug 19, 2016
 */

#include "gtest/gtest.h"
#include "TRROM_UtestUtils.hpp"

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_Radius.hpp"
#include "TRROM_Circle.hpp"
#include "TRROM_Rosenbrock.hpp"
#include "TRROM_SerialArray.hpp"
#include "TRROM_ReducedHessian.hpp"
#include "TRROM_AssemblyMngTypeLP.hpp"
#include "TRROM_TrustRegionNewton.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_InexactNewtonDataMng.hpp"
#include "TRROM_AugmentedLagrangianTypeLP.hpp"
#include "TRROM_AugmentedLagrangianDataMng.hpp"
#include "TRROM_TrustRegionAugmentedLagrangian.hpp"

namespace TrromAugmentedLagrangianTest
{

TEST(AlgorithmKelleySachs, getMin_UsrDefGrad_UsrDefHess_Rosenbrock)
{
    int ncontrols = 2;
    std::tr1::shared_ptr<trrom::Data> data(new trrom::Data);
    trrom::SerialArray<double> control(ncontrols, 2.);
    data->allocateControl(control);
    data->setControlLowerBound(-1e2);
    data->setControlUpperBound(1e2);

    std::tr1::shared_ptr<trrom::Rosenbrock> objective(new trrom::Rosenbrock);
    std::tr1::shared_ptr<trrom::ReducedHessian> hessian(new trrom::ReducedHessian);
    std::tr1::shared_ptr<trrom::AssemblyMngTypeLP> manager(new trrom::AssemblyMngTypeLP(objective));
    std::tr1::shared_ptr<trrom::InexactNewtonDataMng> data_mng(new trrom::InexactNewtonDataMng(data, manager));

    EXPECT_EQ(trrom::types::REDUCED_HESSIAN, hessian->type());
    std::tr1::shared_ptr<trrom::KelleySachsStepMng> step_mng(new trrom::KelleySachsStepMng(data, hessian));
    trrom::TrustRegionNewton algorithm(data, step_mng, data_mng);
    algorithm.getMin();

    EXPECT_EQ(trrom::types::ACTUAL_REDUCTION_TOL_SATISFIED, algorithm.getStoppingCriterion());
    EXPECT_EQ(16, algorithm.getNumOptimizationItrDone());
    data->control()->fill(1.);
    trrom::test::checkResults(*data_mng->getNewPrimal(), *data->control(), 1e-6);
}

TEST(KelleySachsAugmentedLagrangian, getMin_UsrDefGrad_UsrDefHess_Circle)
{
    int ncontrols = 2;
    int num_constraints = 1;
    trrom::SerialArray<double> control(ncontrols, 0.5);
    trrom::SerialArray<double> dual(num_constraints, 0.);
    trrom::SerialArray<double> slacks(num_constraints, 0.);

    std::tr1::shared_ptr<trrom::Data> data(new trrom::Data);
    data->allocateControl(control);
    data->setControlLowerBound(-1e2);
    data->setControlUpperBound(1e2);
    data->allocateDual(dual);
    data->allocateSlacks(slacks);

    std::tr1::shared_ptr<trrom::Circle> objective(new trrom::Circle);
    std::tr1::shared_ptr<trrom::Radius> inequality(new trrom::Radius);
    std::vector<std::tr1::shared_ptr<trrom::InequalityTypeLP> > inequalities(num_constraints, inequality);
    std::tr1::shared_ptr<trrom::AugmentedLagrangianTypeLP> assembly_mng(new trrom::AugmentedLagrangianTypeLP(data, objective, inequalities));
    std::tr1::shared_ptr<trrom::AugmentedLagrangianDataMng> data_mng(new trrom::AugmentedLagrangianDataMng(data, assembly_mng));

    std::tr1::shared_ptr<trrom::ReducedHessian> hessian(new trrom::ReducedHessian);
    std::tr1::shared_ptr<trrom::KelleySachsStepMng> step_mng(new trrom::KelleySachsStepMng(data, hessian));
    trrom::TrustRegionAugmentedLagrangian algorithm(data, step_mng, data_mng);
    algorithm.getMin();

    EXPECT_EQ(trrom::types::OPTIMALITY_AND_FEASIBILITY_SATISFIED, algorithm.getStoppingCriterion());
    EXPECT_EQ(25, algorithm.getNumOptimizationItrDone());
    (*data->control())[0] = 0.31160842594;
    (*data->control())[1] = 0.9503093142;
    trrom::test::checkResults(*data_mng->getNewPrimal(), *data->control(), 1e-6);
}

}
