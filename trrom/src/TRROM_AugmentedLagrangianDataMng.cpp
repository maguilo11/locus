/*
 * TRROM_AugmentedLagrangianDataMng.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_AugmentedLagrangianDataMng.hpp"
#include "TRROM_AugmentedLagrangianAssemblyMng.hpp"

namespace trrom
{

AugmentedLagrangianDataMng::AugmentedLagrangianDataMng(const std::shared_ptr<trrom::Data> & data_,
                                                       const std::shared_ptr<trrom::AugmentedLagrangianAssemblyMng> & mng_) :
        trrom::OptimizationDataMng(data_),
        m_AssemblyMng(mng_)
{
}

AugmentedLagrangianDataMng::~AugmentedLagrangianDataMng()
{
}

void AugmentedLagrangianDataMng::updateInequalityConstraintValues()
{
    m_AssemblyMng->updateInequalityConstraintValues();
}

double AugmentedLagrangianDataMng::evaluateObjective()
{
    ///
    /// Evaluate augmented Lagrangian objective
    ///
    bool objective_inexactness_tolerance_exceeded = false;
    double tolerance = this->getObjectiveInexactnessTolerance();
    double value = m_AssemblyMng->objective(this->getNewPrimal(), tolerance, objective_inexactness_tolerance_exceeded);
    this->setObjectiveInexactnessFlag(objective_inexactness_tolerance_exceeded);
    return (value);
}

double AugmentedLagrangianDataMng::evaluateObjective(const std::shared_ptr<trrom::Vector<double> > & input_)
{
    ///
    /// Evaluate augmented Lagrangian objective
    ///
    bool objective_inexactness_tolerance_exceeded = false;
    double tolerance = this->getObjectiveInexactnessTolerance();
    double value = m_AssemblyMng->objective(input_, tolerance, objective_inexactness_tolerance_exceeded);
    this->setObjectiveInexactnessFlag(objective_inexactness_tolerance_exceeded);
    return (value);
}

int AugmentedLagrangianDataMng::getObjectiveFunctionEvaluationCounter() const
{
    ///
    /// Get augmented Lagrangian evaluation counter
    ///
    return (m_AssemblyMng->getObjectiveCounter());
}

void AugmentedLagrangianDataMng::computeGradient()
{
    ///
    /// Compute augmented Lagrangian gradient
    ///
    bool gradient_inexactness_tol_exceeded = false;
    double tolerance = this->getGradientInexactnessTolerance();
    m_AssemblyMng->gradient(this->getNewPrimal(), this->getNewGradient(), tolerance, gradient_inexactness_tol_exceeded);
}

void AugmentedLagrangianDataMng::computeGradient(const std::shared_ptr<trrom::Vector<double> > & input_,
                                                 const std::shared_ptr<trrom::Vector<double> > & output_)
{
    ///
    /// Compute augmented Lagrangian gradient
    ///
    bool gradient_inexactness_tol_exceeded = false;
    double tolerance = this->getGradientInexactnessTolerance();
    m_AssemblyMng->gradient(input_, output_, tolerance, gradient_inexactness_tol_exceeded);
}

void AugmentedLagrangianDataMng::applyVectorToHessian(const std::shared_ptr<trrom::Vector<double> > & input_,
                                                      const std::shared_ptr<trrom::Vector<double> > & output_)
{
    ///
    /// Apply vector to augmented Lagrangian Hessian
    ///
    bool inexactness_violated = false;
    double tolerance = std::numeric_limits<double>::max();
    m_AssemblyMng->hessian(this->getNewPrimal(), input_, output_, tolerance, inexactness_violated);
}

double AugmentedLagrangianDataMng::getPenalty() const
{
    return (m_AssemblyMng->getPenalty());
}

double AugmentedLagrangianDataMng::getNormInequalityConstraints() const
{
    return (m_AssemblyMng->getNormInequalityConstraints());
}

double AugmentedLagrangianDataMng::getNormLagrangianGradient() const
{
    return (m_AssemblyMng->getNormLagrangianGradient());
}

bool AugmentedLagrangianDataMng::updateLagrangeMultipliers()
{
    bool penalty_below_tolerance = m_AssemblyMng->updateLagrangeMultipliers();
    return (penalty_below_tolerance);
}

}
