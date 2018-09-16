/*
 * DOTk_OptimalityCriteria.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */
#include <cmath>
#include <omp.h>
#include <limits>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_ObjectiveFunction.hpp"
#include "DOTk_EqualityConstraint.hpp"
#include "DOTk_OptimalityCriteria.hpp"
#include "DOTk_InequalityConstraint.hpp"
#include "DOTk_OptimalityCriteriaDataMng.hpp"
#include "DOTk_OptimalityCriteriaRoutineMng.hpp"

namespace dotk
{

DOTk_OptimalityCriteria::DOTk_OptimalityCriteria
(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
 const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
 const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_,
 const std::shared_ptr<dotk::DOTk_InequalityConstraint<Real> > & inequality_) :
        m_LastTime(false),
        m_NumItrDone(0),
        m_PrintDiagnostics(false),
        m_ObjectiveFunctionValue(std::numeric_limits<Real>::max()),
        m_OutputStream(),
        m_DataMng(std::make_shared<dotk::DOTk_OptimalityCriteriaDataMng>(primal_)),
        m_RoutineMng(std::make_shared<dotk::DOTk_OptimalityCriteriaRoutineMng>(objective_, equality_, inequality_))
{
}

DOTk_OptimalityCriteria::~DOTk_OptimalityCriteria()
{
}

size_t DOTk_OptimalityCriteria::getNumItrDone() const
{
    return (m_NumItrDone);
}

bool DOTk_OptimalityCriteria::printDiagnostics() const
{
    return (m_PrintDiagnostics);
}

void DOTk_OptimalityCriteria::enableDiagnostics()
{
    m_PrintDiagnostics = true;
}

Real DOTk_OptimalityCriteria::getInequalityDual() const
{
    return (m_DataMng->getInequalityDual());
}

Real DOTk_OptimalityCriteria::getInequalityConstraintResidual() const
{
    return (m_DataMng->getInequalityConstraintResidual());
}

Real DOTk_OptimalityCriteria::getOptimalObjectiveFunctionValue() const
{
    return (m_DataMng->getNewObjectiveFunctionValue());
}

void DOTk_OptimalityCriteria::setMoveLimit(Real value_)
{
    Real move_limit = value_;
    m_DataMng->setMoveLimit(move_limit);
}

void DOTk_OptimalityCriteria::setDampingParameter(Real value_)
{
    Real damping_parameter = value_;
    m_DataMng->setDampingParameter(damping_parameter);
}

void DOTk_OptimalityCriteria::setGradientTolerance(Real value_)
{
    Real tolerance = value_;
    m_DataMng->setGradientTolerance(tolerance);
}

void DOTk_OptimalityCriteria::setBisectionTolerance(Real value_)
{
    Real tolerance = value_;
    m_DataMng->setBisectionTolerance(tolerance);
}

void DOTk_OptimalityCriteria::setFeasibilityTolerance(Real value_)
{
    Real tolerance = value_;
    m_DataMng->setFeasibilityTolerance(tolerance);
}

void DOTk_OptimalityCriteria::setControlStagnationTolerance(Real value_)
{
    Real tolerance = value_;
    m_DataMng->setControlStagnationTolerance(tolerance);
}

void DOTk_OptimalityCriteria::setInequalityConstraintDualLowerBound(Real value_)
{
    Real lower_bound = value_;
    m_DataMng->setInequalityConstraintDualLowerBound(lower_bound);
}

void DOTk_OptimalityCriteria::setMaxNumOptimizationItr(size_t value_)
{
    size_t max_num_itr = value_;
    m_DataMng->setMaxNumOptimizationItr(max_num_itr);
}

void DOTk_OptimalityCriteria::setInequalityConstraintDualUpperBound(Real value_)
{
    Real upper_bound = value_;
    m_DataMng->setInequalityConstraintDualUpperBound(upper_bound);
}

void DOTk_OptimalityCriteria::gatherSolution(dotk::Vector<Real> & data_) const
{
    data_.update(1., m_DataMng->getNewControl(), 0.);
}

void DOTk_OptimalityCriteria::gatherGradient(dotk::Vector<Real> & data_) const
{
    data_.update(1., m_DataMng->getObjectiveGradient(), 0.);
}

void DOTk_OptimalityCriteria::gatherOuputStream(std::ostringstream & output_)
{
    output_ << m_OutputStream.str().c_str();
}

void DOTk_OptimalityCriteria::getMin()
{
    while(1)
    {
        //m_RoutineMng->applyFilter(m_DataMng);
        m_RoutineMng->solveEqualityConstraint(m_DataMng);
        m_ObjectiveFunctionValue = m_RoutineMng->evaluateObjectiveFunction(m_DataMng);
        m_RoutineMng->computeObjectiveFunctionGradient(m_DataMng);
        this->printCurrentProgress();

        if(m_LastTime == true)
        {
            break;
        }

        this->optimalityCriteriaUpdate();

        this->updateIterationCount();
        m_RoutineMng->computeMaxControlRelativeDifference(m_DataMng);

        if(this->stoppingCriteriaSatisfied() == true)
        {
            m_LastTime = true;
        }
    }
}

void DOTk_OptimalityCriteria::updateIterationCount()
{
    m_NumItrDone++;
}

bool DOTk_OptimalityCriteria::stoppingCriteriaSatisfied()
{
    bool stopping_criterion_satisfied = false;
    if(this->getNumItrDone() > m_DataMng->getMaxNumOptimizationItr())
    {
        stopping_criterion_satisfied = true;
    }
    else if(m_DataMng->getMaxControlRelativeDifference() < m_DataMng->getControlStagnationTolerance())
    {
        stopping_criterion_satisfied = true;
    }
    else if(m_DataMng->getNormObjectiveFunctionGradient() < m_DataMng->getGradientTolerance()
            && m_DataMng->getInequalityConstraintResidual() < m_DataMng->getFeasibilityTolerance())
    {
        stopping_criterion_satisfied = true;
    }

    return (stopping_criterion_satisfied);
}

void DOTk_OptimalityCriteria::printCurrentProgress()
{
    if(this->printDiagnostics() == false)
    {
        return;
    }

    size_t current_itr_count = this->getNumItrDone();

    if(current_itr_count < 2)
    {
        m_OutputStream << " Itr" << std::setw(14) << "   F(x)  " << std::setw(16) << " ||F'(x)||" << std::setw(16)
                << "   H(x)  " << std::setw(16) << "   Res(H) " << "\n" << std::flush;
        m_OutputStream << "-----" << std::setw(14) << "----------" << std::setw(16) << "-----------" << std::setw(16)
                << "----------" << std::setw(16) << "----------" << "\n" << std::flush;
    }

    Real dual = m_DataMng->getInequalityDual();
    Real norm_objective_gradient = m_DataMng->getNormObjectiveFunctionGradient();
    Real inequality_const_residual = m_DataMng->getInequalityConstraintResidual();
    m_OutputStream << std::setw(3) << current_itr_count << std::setprecision(4) << std::fixed << std::scientific
            << std::setw(16) << m_ObjectiveFunctionValue << std::setw(16) << norm_objective_gradient << std::setw(16)
            << dual << std::setw(16) << inequality_const_residual << "\n";
}

void DOTk_OptimalityCriteria::updateControl()
{
    Real value = 0;
    Real new_control = 0;
    Real move_limit = m_DataMng->getMoveLimit();
    Real damping = m_DataMng->getDampingParameter();
    Real multiplier = m_DataMng->getInequalityDual();
    m_RoutineMng->computeInequalityConstraintGradient(m_DataMng);

    size_t num_controls = m_DataMng->getNewControl().size();

    for(size_t index = 0; index < num_controls; ++ index)
    {
        Real old_control = (m_DataMng->getOldControl())[index];
        value = -(m_DataMng->getObjectiveGradient())[index]
                / (multiplier * (m_DataMng->getInequalityGradient())[index]);
        Real fabs_value = std::abs(value);
        Real sign_value = copysign(1.0, value);
        value = old_control * sign_value * std::pow(fabs_value, damping);
        new_control = old_control + move_limit;
        value = std::min(new_control, value);
        value = std::min((m_DataMng->getControlUpperBound())[index], value);
        new_control = old_control - move_limit;
        value = std::max(new_control, value);
        (m_DataMng->getNewControl())[index] = std::max((m_DataMng->getControlLowerBound())[index], value);
    }
}

void DOTk_OptimalityCriteria::optimalityCriteriaUpdate()
{
    Real bisection_tolerance = m_DataMng->getBisectionTolerance();
    Real dual_lower_bound = m_DataMng->getInequalityConstraintDualLowerBound();
    Real dual_upper_bound = m_DataMng->getInequalityConstraintDualUpperBound();

    Real dual_misfit = dual_upper_bound - dual_lower_bound;
    while(dual_misfit >= bisection_tolerance)
    {
        Real dual = static_cast<Real>(0.5) * (dual_upper_bound + dual_lower_bound);
        m_DataMng->setInequalityDual(dual);
        this->updateControl();

        Real residual = m_RoutineMng->computeInequalityConstraintResidual(m_DataMng);

        if(residual > static_cast<Real>(0.))
        {
            dual_lower_bound = dual;
        }
        else
        {
            dual_upper_bound = dual;
        }
        dual_misfit = dual_upper_bound - dual_lower_bound;
    }
}

}
