/*
 * DOTk_LineSearchMngTypeULPTest.cpp
 *
 *  Created on: Aug 25, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"

#include "DOTk_AssemblyManager.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkLineSearchMngTypeULPTest
{

TEST(LineSearchMngTypeULP, getTrialStep)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    EXPECT_EQ(ncontrols, mng.getTrialStep()->size());
    EXPECT_EQ(1, mng.getTrialStep().use_count());
    dotk::StdVector<Real> gold(ncontrols, 0.);
    dotk::gtest::checkResults(*(mng.getTrialStep()), gold);
    dotk::gtest::checkResults(*(mng.getTrialStep()), gold);
}

TEST(LineSearchMngTypeULP, setTrialStep)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    mng.setTrialStep(*primal->control());
    dotk::StdVector<Real> gold(ncontrols, 2.);
    dotk::gtest::checkResults(*(mng.getTrialStep()), gold);
}

TEST(LineSearchMngTypeULP, getHessTimesVec)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    EXPECT_EQ(ncontrols, mng.getMatrixTimesVector()->size());
    EXPECT_EQ(1, mng.getMatrixTimesVector().use_count());
    dotk::StdVector<Real> gold(ncontrols, 0.);
    dotk::gtest::checkResults(*(mng.getMatrixTimesVector()), gold);
}

TEST(LineSearchMngTypeULP, getAndSetObjectiveFunctionValue)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    EXPECT_EQ(0., mng.getOldObjectiveFunctionValue());
    EXPECT_EQ(0., mng.getNewObjectiveFunctionValue());
    mng.setOldObjectiveFunctionValue(static_cast<Real>(1.1));
    EXPECT_EQ(1.1, mng.getOldObjectiveFunctionValue());
    mng.setNewObjectiveFunctionValue(static_cast<Real>(0.9));
    EXPECT_EQ(0.9, mng.getNewObjectiveFunctionValue());
}

TEST(LineSearchMngTypeULP, getPrimal)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    EXPECT_EQ(ncontrols, mng.getOldPrimal()->size());
    EXPECT_EQ(ncontrols, mng.getNewPrimal()->size());
    EXPECT_EQ(1, mng.getOldPrimal().use_count());
    EXPECT_EQ(1, mng.getNewPrimal().use_count());
    dotk::StdVector<Real> gold(ncontrols, 0.);
    dotk::gtest::checkResults(*mng.getOldPrimal(), gold);
    dotk::gtest::checkResults(*mng.getOldPrimal(), gold);
    gold.fill(2.);
    dotk::gtest::checkResults(*mng.getNewPrimal(), gold);
    dotk::gtest::checkResults(*mng.getNewPrimal(), gold);
}

TEST(LineSearchMngTypeULP, setPrimal)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    std::shared_ptr<dotk::Vector<Real> > map = primal->control()->clone();
    map->fill(2);
    mng.setOldPrimal(*map);

    dotk::StdVector<Real> gold(ncontrols, 2.);
    dotk::gtest::checkResults(*(mng.getOldPrimal()), gold);

    mng.setNewPrimal(gold);
    dotk::gtest::checkResults(*(mng.getNewPrimal()), gold);
}

TEST(LineSearchMngTypeULP, getGradient)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    EXPECT_EQ(ncontrols, mng.getOldGradient()->size());
    EXPECT_EQ(ncontrols, mng.getNewGradient()->size());
    EXPECT_EQ(1, mng.getOldGradient().use_count());
    EXPECT_EQ(1, mng.getNewGradient().use_count());
    dotk::StdVector<Real> gold(ncontrols, 0.);
    dotk::gtest::checkResults(*mng.getOldGradient(), gold);
    dotk::gtest::checkResults(*mng.getOldGradient(), gold);
    dotk::gtest::checkResults(*mng.getNewGradient(), gold);
    dotk::gtest::checkResults(*mng.getNewGradient(), gold);
}

TEST(LineSearchMngTypeULP, setGradient)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    dotk::DOTk_LineSearchMngTypeULP mng(primal);

    std::shared_ptr<dotk::Vector<Real> > map = primal->control()->clone();
    map->fill(2);
    mng.setOldGradient(*map);

    dotk::StdVector<Real> gold(ncontrols, 2.);
    dotk::gtest::checkResults(*(mng.getOldGradient()), gold);

    mng.setNewGradient(gold);
    dotk::gtest::checkResults(*(mng.getNewGradient()), gold);
}

TEST(LineSearchMngTypeULP, objective)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    Real tol = 1e-8;
    std::shared_ptr<dotk::Vector<Real> > vector = primal->control()->clone();
    vector->fill(2);
    EXPECT_NEAR(401., mng.getRoutinesMng()->objective(vector), tol);
}

TEST(LineSearchMngTypeULP, Fval_P)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > fval = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > vector = primal->control()->clone();
    std::vector< std::shared_ptr<dotk::Vector<Real> > > control(2, primal->control()->clone());
    control[0]->update(1., *primal->control(), 0.);
    control[1]->update(1., *primal->control(), 0.);
    size_t numvars = 2;
    dotk::StdVector<Real> gold(numvars, 401.);
    mng.getRoutinesMng()->objective(control, fval);

    dotk::gtest::checkResults(*fval, gold);
}

TEST(LineSearchMngTypeULP, Grad)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > grad = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > vector = primal->control()->clone();
    vector->fill(2);
    mng.getRoutinesMng()->gradient(vector, grad);

    dotk::StdVector<Real> gold(ncontrols, 0.);
    gold[0] = 1602.;
    gold[1] = -400.;
    dotk::gtest::checkResults(*grad, gold);
}

TEST(LineSearchMngTypeULP, Hess)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal(new dotk::DOTk_Primal);
    primal->allocateSerialControlArray(ncontrols, 2);
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective(new dotk::DOTk_Rosenbrock);
    dotk::DOTk_LineSearchMngTypeULP mng(primal, objective);

    std::shared_ptr<dotk::Vector<Real> > hess = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > vector = primal->control()->clone();
    std::shared_ptr<dotk::Vector<Real> > trial_step = primal->control()->clone();

    vector->fill(2);
    trial_step->fill(1);
    mng.getRoutinesMng()->hessian(vector, trial_step, hess);

    dotk::StdVector<Real> gold(ncontrols, 0.);
    gold[0] = 3202.;
    gold[1] = -600.;
    dotk::gtest::checkResults(*hess, gold);
}

}
