/*
 * DOTk_ProjLeftPrecCgTest.cpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_MathUtils.hpp"
#include "DOTk_FixedCriterion.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_LeftPreconditioner.hpp"
#include "DOTk_ProjectedLeftPrecCG.hpp"
#include "DOTk_ProjLeftPrecCgDataMng.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"
#include "DOTk_NocedalAndWrightEquality.hpp"
#include "DOTk_NocedalAndWrightObjective.hpp"

namespace DOTkProjLeftPrecCgTest
{

TEST(DOTk_ProjLeftPrecCgTest, checkOrthogonalityMeasure)
{
    size_t nduals = 2;
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    size_t krylov_subspace_dim = 3;
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    dotk::DOTk_ProjectedLeftPrecCG solver(primal, hessian, krylov_subspace_dim);

    (*primal->dual())[0] = 1.;
    (*primal->dual())[1] = 2.;
    solver.getDataMng()->getResidual(0)->dual()->update(1., *primal->dual(), 0.);
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 4.;
    solver.getDataMng()->getResidual(0)->control()->update(1., *primal->control(), 0.);

    (*primal->dual())[0] = 5.;
    (*primal->dual())[1] = 6.;
    solver.getDataMng()->getResidual(1)->dual()->update(1., *primal->dual(), 0.);
    (*primal->control())[0] = 7.;
    (*primal->control())[1] = 8.;
    solver.getDataMng()->getResidual(1)->control()->update(1., *primal->control(), 0.);

    (*primal->dual())[0] = 9.;
    (*primal->dual())[1] = 10.;
    solver.getDataMng()->getResidual(2)->dual()->update(1., *primal->dual(), 0.);
    (*primal->control())[0] = 11.;
    (*primal->control())[1] = 12.;
    solver.getDataMng()->getResidual(2)->control()->update(1., *primal->control(), 0);

    (*primal->dual())[0] = 1.;
    (*primal->dual())[1] = 2.;
    solver.getDataMng()->getLeftPrecTimesVector(0)->dual()->update(1., *primal->dual(), 0.);
    (*primal->control())[0] = 3.;
    (*primal->control())[1] = 4.;
    solver.getDataMng()->getLeftPrecTimesVector(0)->control()->update(1., *primal->control(), 0.);

    (*primal->dual())[0] = 5.;
    (*primal->dual())[1] = 6.;
    solver.getDataMng()->getLeftPrecTimesVector(1)->dual()->update(1., *primal->dual(), 0.);
    (*primal->control())[0] = 7.;
    (*primal->control())[1] = 8.;
    solver.getDataMng()->getLeftPrecTimesVector(1)->control()->update(1., *primal->control(), 0.);

    (*primal->dual())[0] = 9.;
    (*primal->dual())[1] = 10.;
    solver.getDataMng()->getLeftPrecTimesVector(2)->dual()->update(1., *primal->dual(), 0.);
    (*primal->control())[0] = 11.;
    (*primal->control())[1] = 12.;
    solver.getDataMng()->getLeftPrecTimesVector(2)->control()->update(1., *primal->control(), 0.);

    // TEST 0: FIRST ITERATION
    size_t current_itr = 0;
    solver.setNumSolverItrDone(current_itr);
    bool is_orthogonality_measure_valid = solver.checkOrthogonalityMeasure();
    const Real tolerance = 1e-8;
    EXPECT_FALSE(is_orthogonality_measure_valid);
    EXPECT_NEAR(0., solver.getOrthogonalityMeasure(0, 0), tolerance);

    // TEST 1: FIRST ITERATION
    current_itr = 1;
    solver.setNumSolverItrDone(current_itr);
    is_orthogonality_measure_valid = solver.checkOrthogonalityMeasure();
    EXPECT_TRUE(is_orthogonality_measure_valid);
    EXPECT_NEAR(0., solver.getOrthogonalityMeasure(0, 0), tolerance);
    EXPECT_NEAR(0., solver.getOrthogonalityMeasure(0, 0), tolerance);
    EXPECT_NEAR(0.968863931626966, solver.getOrthogonalityMeasure(0, 1), tolerance);
    EXPECT_NEAR(0.968863931626966, solver.getOrthogonalityMeasure(1, 0), tolerance);
    EXPECT_NEAR(0., solver.getOrthogonalityMeasure(1, 1), tolerance);

    // TEST 2: SECOND ITERATION
    current_itr = 2;
    solver.setNumSolverItrDone(current_itr);
    is_orthogonality_measure_valid = solver.checkOrthogonalityMeasure();
    EXPECT_TRUE(is_orthogonality_measure_valid);
    EXPECT_NEAR(0., solver.getOrthogonalityMeasure(0, 0), tolerance);
    EXPECT_NEAR(0.968863931626966, solver.getOrthogonalityMeasure(0, 1), tolerance);
    EXPECT_NEAR(0.950965208670456, solver.getOrthogonalityMeasure(0, 2), tolerance);
    EXPECT_NEAR(0.968863931626966, solver.getOrthogonalityMeasure(1, 0), tolerance);
    EXPECT_NEAR(0., solver.getOrthogonalityMeasure(1, 1), tolerance);
    EXPECT_NEAR(0.997936120806977, solver.getOrthogonalityMeasure(1, 2), tolerance);
    EXPECT_NEAR(0.950965208670456, solver.getOrthogonalityMeasure(2, 0), tolerance);
    EXPECT_NEAR(0.997936120806977, solver.getOrthogonalityMeasure(2, 1), tolerance);
    EXPECT_NEAR(0., solver.getOrthogonalityMeasure(2, 2), tolerance);
}

TEST(DOTk_ProjLeftPrecCgTest, ppcg)
{
    size_t nduals = 3;
    size_t ncontrols = 5;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols);
    (*primal->control())[0] = -1.8;
    (*primal->control())[1] = 1.7;
    (*primal->control())[2] = 1.9;
    (*primal->control())[3] = -0.8;
    (*primal->control())[4] = -0.8;
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

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

    size_t itr = 200;
    primal->dual()->fill(0.);
    primal->control()->fill(0.);
    std::tr1::shared_ptr<dotk::DOTk_Hessian> hessian(new dotk::DOTk_Hessian);
    hessian->setFullSpaceHessian();
    std::tr1::shared_ptr<dotk::DOTk_ProjLeftPrecCgDataMng> smng(new dotk::DOTk_ProjLeftPrecCgDataMng(primal, hessian, itr));
    smng->setAugmentedSystemPrecWithGmresSolver(primal);

    dotk::DOTk_ProjectedLeftPrecCG solver(smng);
    dotk::update(1., mng->getNewGradient(), 0., solver.getDataMng()->getResidual(0));
    dotk::update(1., mng->getHessTimesNormalStep(), 1., solver.getDataMng()->getResidual(0));

    solver.getDataMng()->getLeftPrec()->setParameter(dotk::types::TRUST_REGION_RADIUS, 1e4);
    long double grad_dot_grad = mng->getNewGradient()->dot(*mng->getNewGradient());
    Real grad_norm = std::sqrt(grad_dot_grad);
    solver.getDataMng()->getLeftPrec()->setParameter(dotk::types::NORM_GRADIENT, grad_norm);

    Real tolerance = 1e-12;
    std::tr1::shared_ptr<dotk::DOTk_FixedCriterion> criterion(new dotk::DOTk_FixedCriterion(tolerance));

    solver.ppcg(solver.getDataMng()->getResidual(0), criterion, mng);

    std::tr1::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 0.0196000125684081;
    (*gold)[1] = -0.0219737165126789;
    (*gold)[2] = 0.0361541754328155;
    (*gold)[3] = -0.00246400460771204;
    (*gold)[4] = -0.00246400460771204;
    dotk::gtest::checkResults(*smng->getSolution()->control(), *gold);
    dotk::gtest::checkResults(*smng->getFirstSolution()->control(), *gold);
    (*gold)[0] = -0.00166757862663005;
    (*gold)[1] = 0.00186953451566829;
    (*gold)[2] = -0.00307601487523399;
    (*gold)[3] = 0.000209638713516003;
    (*gold)[4] = 0.000209638713516003;
    dotk::gtest::checkResults(*smng->getResidual(0)->control(), *gold);
}

}
