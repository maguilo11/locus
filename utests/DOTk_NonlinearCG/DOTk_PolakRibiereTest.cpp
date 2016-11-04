/*
 * DOTk_PolakRibiereTest.cpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_Daniels.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_PolakRibiere.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"
#include "DOTk_SecondOrderOperator.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

namespace DOTkNonlinearConjugateGradientTest
{

TEST(DOTk_Daniels, getDirection)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    objective->gradient(*primal->control(), *mng->getNewGradient());
    mng->getTrialStep()->copy(*mng->getNewGradient());

    // SET OLD GRADIENT TO AVOID RETURNING PREMATURELY DUE TO ORTHOGONALITY CHECK
    primal->control()->fill(2.1);
    objective->gradient(*primal->control(), *mng->getOldGradient());

    Real tolerance = 1e-8;
    EXPECT_NEAR(1602., (*mng->getNewGradient())[0], tolerance);
    EXPECT_NEAR(-400., (*mng->getNewGradient())[1], tolerance);

    (*mng->getTrialStep())[0] = -1.;
    (*mng->getTrialStep())[1] = 1.;

    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setForwardDifference(primal);
    hessian->apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());
    EXPECT_NEAR(-4801.9971998201072, (*mng->getMatrixTimesVector())[0], tolerance);
    EXPECT_NEAR(999.99979994436217, (*mng->getMatrixTimesVector())[1], tolerance);

    dotk::DOTk_Daniels nlcg(hessian);
    EXPECT_TRUE(nlcg.getNonlinearCGType() == dotk::types::DANIELS_NLCG);
    nlcg.direction(mng);

    EXPECT_NEAR(-1394.829992917, nlcg.getScaleFactor(), tolerance);
    EXPECT_NEAR(-207.1700070823, (*mng->getTrialStep())[0], tolerance);
    EXPECT_NEAR(-994.8299929176, (*mng->getTrialStep())[1], tolerance);

}

TEST(DOTk_Daniels, direction_TakeSteepestDescent)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng(new dotk::DOTk_LineSearchMngTypeULP(primal, objective));

    objective->gradient(*primal->control(), *mng->getNewGradient());
    mng->getTrialStep()->copy(*mng->getNewGradient());

    Real tolerance = 1e-8;
    EXPECT_NEAR(1602., (*mng->getNewGradient())[0], tolerance);
    EXPECT_NEAR(-400., (*mng->getNewGradient())[1], tolerance);

    (*mng->getTrialStep())[0] = -1.;
    (*mng->getTrialStep())[1] = 1.;

    std::tr1::shared_ptr<dotk::NumericallyDifferentiatedHessian> hessian(new dotk::NumericallyDifferentiatedHessian(primal, objective));
    hessian->setForwardDifference(primal);
    dotk::DOTk_Daniels nlcg(hessian);
    nlcg.direction(mng);

    EXPECT_NEAR(0, (*mng->getMatrixTimesVector())[1], tolerance);
    EXPECT_NEAR(0, (*mng->getMatrixTimesVector())[0], tolerance);

    EXPECT_NEAR(0, nlcg.getScaleFactor(), tolerance);
    EXPECT_NEAR(-1602, (*mng->getTrialStep())[0], tolerance);
    EXPECT_NEAR(400, (*mng->getTrialStep())[1], tolerance);

}

TEST(DOTk_PolakRibiere, computeScaleFactor)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> operators(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, operators);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng.setNewGradient(*primal->control());

    dotk::DOTk_PolakRibiere dir;
    EXPECT_EQ(dotk::types::POLAK_RIBIERE_NLCG, dir.getNonlinearCGType());
    Real value = dir.computeScaleFactor(mng.getOldGradient(), mng.getNewGradient());
    Real tol = 1e-8;
    EXPECT_NEAR(114.4, value, tol);
}

TEST(DOTk_PolakRibiere, getDirection)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> operators(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, operators);

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng.setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng.setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng.setTrialStep(*primal->control());

    dotk::DOTk_PolakRibiere dir;
    EXPECT_EQ(dotk::types::POLAK_RIBIERE_NLCG, dir.getNonlinearCGType());
    dir.getDirection(mng.getOldGradient(), mng.getNewGradient(), mng.getTrialStep());

    Real tol = 1e-8;
    std::tr1::shared_ptr<dotk::vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -103.4;
    (*gold)[1] = -250.8;
    dotk::gtest::checkResults(*mng.getTrialStep(), *gold);
    EXPECT_NEAR(114.4, dir.getScaleFactor(), tol);
}

TEST(DOTk_PolakRibiere, direction)
{
    size_t ncontrols = 2;
    std::tr1::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::tr1::shared_ptr<dotk::DOTk_Rosenbrock> operators(new dotk::DOTk_Rosenbrock);
    std::tr1::shared_ptr<dotk::DOTk_LineSearchMngTypeULP>
    mng(new dotk::DOTk_LineSearchMngTypeULP(primal, operators));

    (*primal->control())[0] = 1.;
    (*primal->control())[1] = 2.;
    mng->setOldGradient(*primal->control());
    (*primal->control())[0] = -11.;
    (*primal->control())[1] = 22.;
    mng->setNewGradient(*primal->control());
    (*primal->control())[0] = -1.;
    (*primal->control())[1] = -2;
    mng->setTrialStep(*primal->control());

    dotk::DOTk_PolakRibiere dir;
    EXPECT_EQ(dotk::types::POLAK_RIBIERE_NLCG, dir.getNonlinearCGType());
    dir.direction(mng);

    std::tr1::shared_ptr<dotk::vector<Real> > gold = dotk::gtest::allocateControl();
    (*gold)[0] = -103.4;
    (*gold)[1] = -250.8;
    Real tol = 1e-8;
    dotk::gtest::checkResults(*mng->getTrialStep(), *gold);
    EXPECT_NEAR(114.4, dir.getScaleFactor(), tol);
}

}
