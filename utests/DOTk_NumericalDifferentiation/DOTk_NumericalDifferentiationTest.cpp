/*
 * DOTk_NumericalDifferentiationTest.cpp
 *
 *  Created on: Jan 24, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "gtest/gtest.h"

#include "DOTk_Primal.hpp"
#include "DOTk_Rosenbrock.hpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_ObjectiveTypeLP.hpp"
#include "DOTk_LineSearchMngTypeULP.hpp"
#include "DOTk_NumericallyDifferentiatedHessian.hpp"

#include "DOTk_GtestDOTkVecTools.hpp"

namespace DOTkNumericalIntegrationTest
{

TEST(NumericalDerivative, ForwardFiniteDifference)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(num_controls, 2);

    Real tValue = 0;
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr< dotk::Vector<Real> > current_gradient = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    objective->gradient(*primal->control(), *current_gradient);

    dotk::NumericallyDifferentiatedHessian hessian(primal, objective);
    hessian.setForwardDifference(*primal->control());

    Real tDirectionValue = 0.1;
    std::shared_ptr< dotk::Vector<Real> > output = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    std::shared_ptr< dotk::Vector<Real> > direction = std::make_shared<dotk::StdVector<Real>>(num_controls, tDirectionValue);
    hessian.apply(primal->control(), current_gradient, direction, output);

    // Check results with gold values (i.e. true hessian)
    dotk::lp::ObjectiveFunctionSecondDerivative true_hessian(dotk::types::variable_t::CONTROL);
    std::shared_ptr< dotk::Vector<Real> > gold = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    true_hessian(objective, primal->control(), direction, gold);

    Real tolerance = 5e-5;
    dotk::gtest::checkResults(*output, *gold, tolerance);
}

TEST(NumericalDerivative, BackwardFiniteDifference)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(num_controls, 2);

    Real tValue = 0;
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr< dotk::Vector<Real> > current_gradient = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    objective->gradient(*primal->control(), *current_gradient);

    // DEFAULT = BACKWARD DIFFERENCE
    Real tDirectionValue = 0.1;
    dotk::NumericallyDifferentiatedHessian hessian(primal, objective);
    std::shared_ptr< dotk::Vector<Real> > output = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    std::shared_ptr< dotk::Vector<Real> > direction = std::make_shared<dotk::StdVector<Real>>(num_controls, tDirectionValue);
    hessian.apply(primal->control(), current_gradient, direction, output);

    dotk::lp::ObjectiveFunctionSecondDerivative true_hessian(dotk::types::variable_t::CONTROL);
    std::shared_ptr< dotk::Vector<Real> > gold = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    true_hessian(objective, primal->control(), direction, gold);

    Real tolerance = 5e-5;
    dotk::gtest::checkResults(*output, *gold, tolerance);
}

TEST(NumericalDerivative, CentralFiniteDifference)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(num_controls, 2);

    Real tValue = 0;
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr< dotk::Vector<Real> > current_gradient = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    objective->gradient(*primal->control(), *current_gradient);

    dotk::NumericallyDifferentiatedHessian hessian(primal, objective);
    hessian.setCentralDifference(*primal->control());

    Real tDirectionValue = 0.1;
    std::shared_ptr< dotk::Vector<Real> > output = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    std::shared_ptr< dotk::Vector<Real> > direction = std::make_shared<dotk::StdVector<Real>>(num_controls, tDirectionValue);
    hessian.apply(primal->control(), current_gradient, direction, output);

    dotk::lp::ObjectiveFunctionSecondDerivative true_hessian(dotk::types::variable_t::CONTROL);
    std::shared_ptr< dotk::Vector<Real> > gold = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    true_hessian(objective, primal->control(),direction, gold);

    Real tolerance = 1e-6;
    dotk::gtest::checkResults(*output, *gold, tolerance);
}

TEST(NumericalDerivative, SecondOrderForwardFiniteDifference)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(num_controls, 2);

    Real tValue = 0;
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr< dotk::Vector<Real> > current_gradient = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    objective->gradient(*primal->control(), *current_gradient);

    dotk::NumericallyDifferentiatedHessian hessian(primal, objective);
    hessian.setSecondOrderForwardDifference(*primal->control());

    Real tDirectionValue = 0.1;
    std::shared_ptr< dotk::Vector<Real> > direction = std::make_shared<dotk::StdVector<Real>>(num_controls, tDirectionValue);
    std::shared_ptr< dotk::Vector<Real> > output = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    hessian.apply(primal->control(), current_gradient, direction, output);

    dotk::lp::ObjectiveFunctionSecondDerivative true_hessian(dotk::types::variable_t::CONTROL);
    std::shared_ptr< dotk::Vector<Real> > gold = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    true_hessian(objective, primal->control(),direction, gold);

    Real tolerance = 5e-6;
    dotk::gtest::checkResults(*output, *gold, tolerance);
}

TEST(NumericalDerivative, ThirdOrderForwardFiniteDifference)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(num_controls, 2);

    Real tValue = 0;
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr< dotk::Vector<Real> > current_gradient = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    objective->gradient(*primal->control(), *current_gradient);

    dotk::NumericallyDifferentiatedHessian hessian(primal, objective);
    hessian.setThirdOrderForwardDifference(*primal->control());

    Real tDirectionValue = 0.1;
    std::shared_ptr< dotk::Vector<Real> > direction = std::make_shared<dotk::StdVector<Real>>(num_controls, tDirectionValue);
    std::shared_ptr< dotk::Vector<Real> > output = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    hessian.apply(primal->control(), current_gradient, direction, output);

    dotk::lp::ObjectiveFunctionSecondDerivative true_hessian(dotk::types::variable_t::CONTROL);
    std::shared_ptr< dotk::Vector<Real> > gold = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    true_hessian(objective, primal->control(),direction, gold);

    Real tolerance = 1e-6;
    dotk::gtest::checkResults(*output, *gold, tolerance);
}

TEST(NumericalDerivative, ThirdOrderBackwardFiniteDifference)
{
    size_t num_controls = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(num_controls, 2);

    Real tValue = 0;
    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr< dotk::Vector<Real> > current_gradient = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    objective->gradient(*primal->control(), *current_gradient);

    dotk::NumericallyDifferentiatedHessian hessian(primal, objective);
    hessian.setThirdOrderBackwardDifference(*primal->control());

    Real tDirectionValue = 0.1;
    std::shared_ptr< dotk::Vector<Real> > output = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    std::shared_ptr< dotk::Vector<Real> > direction = std::make_shared<dotk::StdVector<Real>>(num_controls, tDirectionValue);
    hessian.apply(primal->control(), current_gradient, direction, output);

    dotk::lp::ObjectiveFunctionSecondDerivative true_hessian(dotk::types::variable_t::CONTROL);
    std::shared_ptr< dotk::Vector<Real> > gold = std::make_shared<dotk::StdVector<Real>>(num_controls, tValue);
    true_hessian(objective, primal->control(),direction, gold);

    Real tolerance = 5e-7;
    dotk::gtest::checkResults(*output, *gold, tolerance);
}

TEST(NumericalDerivative, apply)
{
    size_t ncontrols = 2;
    std::shared_ptr<dotk::DOTk_Primal> primal = std::make_shared<dotk::DOTk_Primal>();
    primal->allocateSerialControlArray(ncontrols, 2);

    std::shared_ptr<dotk::DOTk_Rosenbrock> objective = std::make_shared<dotk::DOTk_Rosenbrock>();
    std::shared_ptr<dotk::DOTk_LineSearchMngTypeULP> mng = std::make_shared<dotk::DOTk_LineSearchMngTypeULP>(primal, objective);

    dotk::NumericallyDifferentiatedHessian hessian(primal, objective);
    hessian.setCentralDifference(*primal->control());

    (*mng->getTrialStep())[0] = 0.1;
    (*mng->getTrialStep())[1] = 0.1;
    (*mng->getNewGradient())[0] = 1602.;
    (*mng->getNewGradient())[1] = -400.;
    hessian.apply(mng, mng->getTrialStep(), mng->getMatrixTimesVector());

    std::shared_ptr<dotk::Vector<Real> > gold = primal->control()->clone();
    (*gold)[0] = 320.2;
    (*gold)[1] = -60.;
    Real tolerance = 1e-6;
    dotk::gtest::checkResults(*gold, *mng->getMatrixTimesVector(), tolerance);
}

}
