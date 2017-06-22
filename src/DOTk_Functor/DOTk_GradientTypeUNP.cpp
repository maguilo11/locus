/*
 * DOTk_GradientTypeUNP.cpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_GradientTypeUNP.hpp"

namespace dotk
{

DOTk_GradientTypeUNP::DOTk_GradientTypeUNP(const std::shared_ptr<dotk::DOTk_Primal> & primal_,
                                           const std::shared_ptr<dotk::DOTk_ObjectiveFunction<Real> > & objective_,
                                           const std::shared_ptr<dotk::DOTk_EqualityConstraint<Real> > & equality_) :
        dotk::DOTk_Functor::DOTk_Functor(dotk::types::GRADIENT_TYPE_UNP),
        m_Dual(),
        m_State(),
        m_StateWorkVec(),
        m_ControlWorkVec(),
        m_ObjectiveFunction(objective_),
        m_EqualityContraint(equality_)
{
    this->initialize(primal_);
}

DOTk_GradientTypeUNP::~DOTk_GradientTypeUNP()
{
}

void DOTk_GradientTypeUNP::operator()(const dotk::Vector<Real> & primal_, dotk::Vector<Real> & gradient_)
{
    // evaluate equality constraint
    m_State->fill(static_cast<Real>(0.0));
    m_EqualityContraint->solve(primal_, *m_State);

    m_StateWorkVec->fill(static_cast<Real>(0.0));
    m_ObjectiveFunction->partialDerivativeState(*m_State, primal_, *m_StateWorkVec);
    m_StateWorkVec->scale(static_cast<Real>(-1.0));

    // solve adjoint problem
    m_Dual->fill(static_cast<Real>(0.0));
    m_EqualityContraint->applyAdjointInverseJacobianState(*m_State, primal_, *m_StateWorkVec, *m_Dual);

    // get equality constraint contribution to the gradient operator
    m_ControlWorkVec->fill(static_cast<Real>(0.0));
    m_EqualityContraint->adjointPartialDerivativeControl(*m_State, primal_, *m_Dual, *m_ControlWorkVec);

    // assemble gradient operator
    gradient_.update(1., *m_ControlWorkVec, 0.);
    m_ControlWorkVec->fill(static_cast<Real>(0.0));
    m_ObjectiveFunction->partialDerivativeControl((*m_State), primal_, *m_ControlWorkVec);
    gradient_.update(static_cast<Real>(1.0), *m_ControlWorkVec, 1.);
}

void DOTk_GradientTypeUNP::initialize(const std::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->dual().use_count() > 0)
    {
        this->allocate(dotk::types::DUAL, primal_->dual());
    }
    else
    {
        std::perror("\n**** Error in DOTk_RoutinesNumIntgHessTypeUNP::initialize. User did not define dual data. ABORT. ****\n");
        std::abort();
    }
    if(primal_->control().use_count() > 0)
    {
        this->allocate(dotk::types::CONTROL, primal_->control());
    }
    else
    {
        std::perror("\n**** Error in DOTk_RoutinesNumIntgHessTypeUNP::initialize. User did not define control data. ABORT. ****\n");
        std::abort();
    }
}

void DOTk_GradientTypeUNP::allocate(dotk::types::variable_t type_, const std::shared_ptr<dotk::Vector<Real> > & data_)
{
    switch(type_)
    {
        case dotk::types::CONTROL:
        {
            m_ControlWorkVec = data_->clone();
            break;
        }
        case dotk::types::DUAL:
        {
            m_Dual = data_->clone();
            m_State = data_->clone();
            m_StateWorkVec = data_->clone();
            break;
        }
        case dotk::types::STATE:
        case dotk::types::PRIMAL:
        case dotk::types::UNDEFINED_VARIABLE:
        {
            break;
        }
    }
}

}
