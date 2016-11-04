/*
 * DOTk_KrylovSolverStoppingCriterion.cpp
 *
 *  Created on: Apr 11, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

DOTk_KrylovSolverStoppingCriterion::DOTk_KrylovSolverStoppingCriterion(dotk::types::stopping_criterion_t type_) :
        m_Type(type_),
        m_Parameters()
{
}

DOTk_KrylovSolverStoppingCriterion::~DOTk_KrylovSolverStoppingCriterion()
{
}

dotk::types::stopping_criterion_t DOTk_KrylovSolverStoppingCriterion::type() const
{
    return (m_Type);
}

void DOTk_KrylovSolverStoppingCriterion::insert(dotk::types::stopping_criterion_param_t type_, Real value_)
{
    m_Parameters.insert(std::pair<dotk::types::stopping_criterion_param_t, Real>(type_, value_));
}

void DOTk_KrylovSolverStoppingCriterion::set(dotk::types::stopping_criterion_param_t type_, Real value_)
{
    m_Parameters.find(type_)->second = value_;
}

Real DOTk_KrylovSolverStoppingCriterion::get(dotk::types::stopping_criterion_param_t type_) const
{
    return (m_Parameters.find(type_)->second);
}

}
