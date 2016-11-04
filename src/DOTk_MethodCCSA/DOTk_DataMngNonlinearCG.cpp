/*
 * DOTk_DataMngNonlinearCG.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <limits>

#include "vector.hpp"
#include "DOTk_DataMngNonlinearCG.hpp"

namespace dotk
{

DOTk_DataMngNonlinearCG::DOTk_DataMngNonlinearCG(const std::tr1::shared_ptr<dotk::vector<Real> > & dual_) :
        m_NewObjectiveFunctionValue(std::numeric_limits<Real>::max()),
        m_OldObjectiveFunctionValue(std::numeric_limits<Real>::max()),
        m_NewDual(dual_->clone()),
        m_OldDual(dual_->clone()),
        m_NewGradient(dual_->clone()),
        m_OldGradient(dual_->clone()),
        m_NewTrialStep(dual_->clone()),
        m_OldTrialStep(dual_->clone()),
        m_NewSteepestDescent(dual_->clone()),
        m_OldSteepestDescent(dual_->clone())
{
}

DOTk_DataMngNonlinearCG::~DOTk_DataMngNonlinearCG()
{
}

void DOTk_DataMngNonlinearCG::reset()
{
    m_NewDual->fill(0);
    m_OldDual->fill(0);
    m_NewGradient->fill(0);
    m_OldGradient->fill(0);
    m_NewTrialStep->fill(0);
    m_OldTrialStep->fill(0);
    m_NewSteepestDescent->fill(0);
    m_OldSteepestDescent->fill(0);

    m_NewObjectiveFunctionValue = 0;
    m_OldObjectiveFunctionValue = 0;
}

void DOTk_DataMngNonlinearCG::storeCurrentState()
{
    m_OldDual->copy(*m_NewDual);
    m_OldGradient->copy(*m_NewGradient);
    m_OldTrialStep->copy(*m_NewTrialStep);
    m_OldSteepestDescent->copy(*m_NewSteepestDescent);

    m_OldObjectiveFunctionValue = m_NewObjectiveFunctionValue;
}

}
