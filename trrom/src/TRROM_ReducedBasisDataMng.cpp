/*
 * TRROM_ReducedBasisDataMng.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Vector.hpp"
#include "TRROM_ReducedBasisData.hpp"
#include "TRROM_ReducedBasisDataMng.hpp"
#include "TRROM_ReducedBasisAssemblyMng.hpp"

namespace trrom
{

ReducedBasisDataMng::ReducedBasisDataMng(const std::tr1::shared_ptr<trrom::ReducedBasisData> & data_,
                                         const std::tr1::shared_ptr<trrom::ReducedBasisAssemblyMng> & manager_) :
        trrom::OptimizationDataMng(data_),
        m_AssemblyMng(manager_)
{
}

ReducedBasisDataMng::~ReducedBasisDataMng()
{
}

void ReducedBasisDataMng::computeGradient()
{
    bool gradient_inexactness_tol_exceeded = false;
    double tolerance = this->getGradientInexactnessTolerance();
    m_AssemblyMng->gradient(this->getNewPrimal(), this->getNewGradient(), tolerance, gradient_inexactness_tol_exceeded);
    if(gradient_inexactness_tol_exceeded == true)
    {
        this->updateLowFidelityModel();
        m_AssemblyMng->gradient(this->getNewPrimal(),
                                this->getNewGradient(),
                                tolerance,
                                gradient_inexactness_tol_exceeded);
    }
}

void ReducedBasisDataMng::computeGradient(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                                          const std::tr1::shared_ptr<trrom::Vector<double> > & output_)
{
    bool gradient_inexactness_tol_exceeded = false;
    double tolerance = this->getGradientInexactnessTolerance();
    m_AssemblyMng->gradient(input_, output_, tolerance, gradient_inexactness_tol_exceeded);
    if(gradient_inexactness_tol_exceeded == true)
    {
        this->updateLowFidelityModel();
        m_AssemblyMng->gradient(input_, output_, tolerance, gradient_inexactness_tol_exceeded);
    }
}

double ReducedBasisDataMng::evaluateObjective()
{
    bool objective_inexactness_tolerance_exceeded = false;
    double tolerance = this->getObjectiveInexactnessTolerance();
    double value = m_AssemblyMng->objective(this->getNewPrimal(), tolerance, objective_inexactness_tolerance_exceeded);
    this->setObjectiveInexactnessFlag(objective_inexactness_tolerance_exceeded);
    return (value);
}

double ReducedBasisDataMng::evaluateObjective(const std::tr1::shared_ptr<trrom::Vector<double> > & input_)
{
    bool objective_inexactness_tolerance_exceeded = false;
    double tolerance = this->getObjectiveInexactnessTolerance();
    double value = m_AssemblyMng->objective(input_, tolerance, objective_inexactness_tolerance_exceeded);
    this->setObjectiveInexactnessFlag(objective_inexactness_tolerance_exceeded);
    return (value);
}

void ReducedBasisDataMng::applyVectorToHessian(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                                               const std::tr1::shared_ptr<trrom::Vector<double> > & output_)
{
    bool inexactness_violated = false;
    double tolerance = std::numeric_limits<double>::max();
    m_AssemblyMng->hessian(this->getNewPrimal(), input_, output_, tolerance, inexactness_violated);
}

int ReducedBasisDataMng::getObjectiveFunctionEvaluationCounter() const
{
    return (m_AssemblyMng->getObjectiveCounter());
}

void ReducedBasisDataMng::updateLowFidelityModel()
{
    m_AssemblyMng->updateLowFidelityModel();
    m_AssemblyMng->fidelity(trrom::types::LOW_FIDELITY);
}

trrom::types::fidelity_t ReducedBasisDataMng::fidelity() const
{
    return (m_AssemblyMng->fidelity());
}

void ReducedBasisDataMng::fidelity(trrom::types::fidelity_t input_)
{
    m_AssemblyMng->fidelity(input_);
}

}
