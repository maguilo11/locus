/*
 * TROOM_AssemblyMngTypeLP.cpp
 *
 *  Created on: Aug 14, 2016
 */

#include "TRROM_Vector.hpp"
#include "TRROM_ObjectiveTypeLP.hpp"
#include "TRROM_AssemblyMngTypeLP.hpp"

namespace trrom
{

AssemblyMngTypeLP::AssemblyMngTypeLP(const std::shared_ptr<trrom::ObjectiveTypeLP> & objective_) :
        m_HessianCounter(0),
        m_GradientCounter(0),
        m_ObjectiveCounter(0),
        m_Objective(objective_)
{
}

AssemblyMngTypeLP::~AssemblyMngTypeLP()
{
}

int AssemblyMngTypeLP::getHessianCounter() const
{
    return (m_HessianCounter);
}

void AssemblyMngTypeLP::updateHessianCounter()
{
    m_HessianCounter++;
}

int AssemblyMngTypeLP::getGradientCounter() const
{
    return (m_GradientCounter);
}

void AssemblyMngTypeLP::updateGradientCounter()
{
    m_GradientCounter++;
}

int AssemblyMngTypeLP::getObjectiveCounter() const
{
    return (m_ObjectiveCounter);
}

void AssemblyMngTypeLP::updateObjectiveCounter()
{
    m_ObjectiveCounter++;
}

double AssemblyMngTypeLP::objective(const std::shared_ptr<trrom::Vector<double> > & input_,
                                    const double & tolerance_,
                                    bool & inexactness_violated_)
{
    double value = m_Objective->value(tolerance_, *input_, inexactness_violated_);
    this->updateObjectiveCounter();
    return (value);
}

void AssemblyMngTypeLP::gradient(const std::shared_ptr<trrom::Vector<double> > & input_,
                                 const std::shared_ptr<trrom::Vector<double> > & gradient_,
                                 const double & tolerance_,
                                 bool & inexactness_violated_)
{
    m_Objective->gradient(tolerance_, *input_, *gradient_, inexactness_violated_);
    this->updateGradientCounter();
}

void AssemblyMngTypeLP::hessian(const std::shared_ptr<trrom::Vector<double> > & input_,
                                const std::shared_ptr<trrom::Vector<double> > & vector_,
                                const std::shared_ptr<trrom::Vector<double> > & hess_times_vec_,
                                const double & tolerance_,
                                bool & inexactness_violated_)
{
    m_Objective->hessian(tolerance_, *input_, *vector_, *hess_times_vec_, inexactness_violated_);
    this->updateHessianCounter();
}

}
