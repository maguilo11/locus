/*
 * DOTk_PrecGMRESTest.cpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_PrecGMRES.hpp"

#include "DOTk_NocedalAndWrightEquality.hpp"
#include "DOTk_NocedalAndWrightObjective.hpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_SerialVector.cpp"
#include "DOTk_GmresTestMatrix.hpp"
#include "DOTk_RelativeCriterion.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_PrecGenMinResDataMng.hpp"
#include "DOTk_TrustRegionMngTypeELP.hpp"

namespace DOTkPrecGMRESTest
{

TEST(DOTk_GmresTestMatrix, apply)
{
    std::tr1::shared_ptr<dotk::vector<Real> > dual = dotk::gtest::allocateData(2, 1);
    std::tr1::shared_ptr<dotk::vector<Real> > control = dotk::gtest::allocateData(2, 1);
    std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> > primal(new dotk::DOTk_MultiVector<Real>(*control, *dual));

    std::tr1::shared_ptr<dotk::vector<Real> > matrix_times_primal = primal->clone();
    dotk::DOTk_GmresTestMatrix matrix(primal);
    matrix.apply(primal, matrix_times_primal);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = primal->clone();
    (*gold->dual())[0] = 1.794823519789998;
    (*gold->dual())[1] = 2.598897110484027;
    (*gold->control())[0] = 1.195922379007526;
    (*gold->control())[1] = 0.867423979503765;
    dotk::gtest::checkResults(*matrix_times_primal, *gold);
}

TEST(DOTk_PrecGMRES, initialize)
{
    size_t nduals = 2;
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));

    std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> >
        multi_vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    multi_vector->fill(1);
    std::tr1::shared_ptr<dotk::DOTk_GmresTestMatrix> matrix(new dotk::DOTk_GmresTestMatrix(multi_vector));
    std::tr1::shared_ptr<dotk::DOTk_PrecGenMinResDataMng> solver_mng(new dotk::DOTk_PrecGenMinResDataMng(primal, matrix));

    Real relative_tolerance = 1e-2;
    std::tr1::shared_ptr<dotk::DOTk_RelativeCriterion> criterion(new dotk::DOTk_RelativeCriterion(relative_tolerance));
    dotk::DOTk_PrecGMRES solver(solver_mng);
    solver.initialize(multi_vector, criterion, mng);

    dotk::gtest::checkResults(*solver.getDataMng()->getResidual(), *multi_vector);
    dotk::gtest::checkResults(*solver.getDataMng()->getLeftPrecTimesVector(), *solver.getDataMng()->getResidual());

    Real tol = 1e-8;
    EXPECT_NEAR(2., solver.getSolverResidualNorm(), tol);
    EXPECT_NEAR(0.02, solver.getInitialStoppingTolerance(), tol);
}

TEST(DOTk_PrecGMRES, gmres)
{
    size_t nduals = 2;
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialDualArray(nduals);
    primal->allocateSerialControlArray(ncontrols, 2.);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightEquality> equality(new dotk::DOTk_NocedalAndWrightEquality);
    std::tr1::shared_ptr<dotk::DOTk_NocedalAndWrightObjective> objective(new dotk::DOTk_NocedalAndWrightObjective);
    std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP>
        mng(new dotk::DOTk_TrustRegionMngTypeELP(primal, objective, equality));


    Real relative_tolerance = 1e-2;
    std::tr1::shared_ptr<dotk::DOTk_RelativeCriterion> criterion(new dotk::DOTk_RelativeCriterion(relative_tolerance));
    std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> >
        multi_vector(new dotk::DOTk_MultiVector<Real>(*primal->control(), *primal->dual()));
    (*multi_vector->dual())[0] = 0.930436494727822;
    (*multi_vector->dual())[1] = 0.846166890508573;
    (*multi_vector->control())[0] = 0.686772712360496;
    (*multi_vector->control())[1] = 0.588976642856829;

    std::tr1::shared_ptr<dotk::DOTk_GmresTestMatrix> matrix(new dotk::DOTk_GmresTestMatrix(multi_vector));
    std::tr1::shared_ptr<dotk::DOTk_PrecGenMinResDataMng> solver_mng(new dotk::DOTk_PrecGenMinResDataMng(primal, matrix));
    EXPECT_EQ(dotk::types::USER_DEFINED_MATRIX, solver_mng->getLinearOperator()->type());

    dotk::DOTk_PrecGMRES solver(solver_mng);
    solver.gmres(multi_vector, criterion, mng);

    std::tr1::shared_ptr<dotk::vector<Real> > control_gold = primal->control()->clone();
    (*control_gold)[0] = 4.959572514354128;
    (*control_gold)[1] = -0.273403699946782;
    dotk::gtest::checkResults(*solver_mng->getSolution()->control(), *control_gold);

    std::tr1::shared_ptr<dotk::vector<Real> > dual_gold = primal->dual()->clone();
    (*dual_gold)[0] = -4.563421968790186;
    (*dual_gold)[1] = 1.837364600836567;
    dotk::gtest::checkResults(*solver_mng->getSolution()->dual(), *dual_gold);
    EXPECT_EQ(4, solver.getNumSolverItrDone());
}

}
