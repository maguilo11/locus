/*
 * TRROM_InexactNewtonDataMng.cpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_AssemblyManager.hpp"
#include "TRROM_InexactNewtonDataMng.hpp"

namespace trrom
{

InexactNewtonDataMng::InexactNewtonDataMng(const std::tr1::shared_ptr<trrom::Data> & data_,
                                           const std::tr1::shared_ptr<trrom::AssemblyManager> & manager_) :
        trrom::OptimizationDataMng(data_),
        m_Data(data_),
        m_AssemblyMng(manager_)
{
}

InexactNewtonDataMng::~InexactNewtonDataMng()
{
}

void InexactNewtonDataMng::computeGradient()
{
    bool gradient_inexactness_tol_exceeded = false;
    double tolerance = this->getGradientInexactnessTolerance();
    m_AssemblyMng->gradient(this->getNewPrimal(), this->getNewGradient(), tolerance, gradient_inexactness_tol_exceeded);
}

void InexactNewtonDataMng::computeGradient(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                                           const std::tr1::shared_ptr<trrom::Vector<double> > & output_)
{
    bool gradient_inexactness_tol_exceeded = false;
    double tolerance = this->getGradientInexactnessTolerance();
    m_AssemblyMng->gradient(input_, output_, tolerance, gradient_inexactness_tol_exceeded);
}

double InexactNewtonDataMng::evaluateObjective()
{
    bool objective_inexactness_tolerance_exceeded = false;
    double tolerance = this->getObjectiveInexactnessTolerance();
    double value = m_AssemblyMng->objective(this->getNewPrimal(), tolerance, objective_inexactness_tolerance_exceeded);
    this->setObjectiveInexactnessFlag(objective_inexactness_tolerance_exceeded);
    return (value);
}

double InexactNewtonDataMng::evaluateObjective(const std::tr1::shared_ptr<trrom::Vector<double> > & input_)
{
    bool objective_inexactness_tolerance_exceeded = false;
    double tolerance = this->getObjectiveInexactnessTolerance();
    double value = m_AssemblyMng->objective(input_, tolerance, objective_inexactness_tolerance_exceeded);
    this->setObjectiveInexactnessFlag(objective_inexactness_tolerance_exceeded);
    return (value);
}

void InexactNewtonDataMng::applyVectorToHessian(const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                                                const std::tr1::shared_ptr<trrom::Vector<double> > & output_)
{
    bool inexactness_violated = false;
    double tolerance = std::numeric_limits<double>::max();
    m_AssemblyMng->hessian(this->getNewPrimal(), input_, output_, tolerance, inexactness_violated);
}

int InexactNewtonDataMng::getObjectiveFunctionEvaluationCounter() const
{
    return (m_AssemblyMng->getObjectiveCounter());
}

const std::tr1::shared_ptr<trrom::Data> & InexactNewtonDataMng::getData() const
{
    return (m_Data);
}

}
