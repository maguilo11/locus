/*
 * DOTk_UtilsCCSA.cpp
 *
 *  Created on: Dec 13, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <algorithm>

#include "vector.hpp"
#include "matrix.hpp"
#include "DOTk_UtilsCCSA.hpp"
#include "DOTk_DataMngCCSA.hpp"

namespace dotk
{

namespace ccsa
{

Real computeResidualNorm(const std::tr1::shared_ptr<dotk::Vector<Real> > & primal_,
                         const std::tr1::shared_ptr<dotk::Vector<Real> > & dual_,
                         const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_)
{
    data_mng_->m_CurrentInequalityGradients->matVec(*dual_, *data_mng_->m_WorkVector, true);
    // Compute optimality contribution to residual
    Real optimality_contribution_1 = 0;
    Real optimality_contribution_2 = 0;
    size_t number_primals = primal_->size();
    for(size_t index = 0; index < number_primals; ++index)
    {
        Real value_one = (*data_mng_->m_CurrentObjectiveGradient)[index] + (*data_mng_->m_WorkVector)[index];
        Real value_two = (static_cast<Real>(1.) + (*primal_)[index]) * std::max(0., value_one);
        optimality_contribution_1 += value_two * value_two;
        value_two = (static_cast<Real>(1.) - (*primal_)[index]) * std::max(0., -value_one);
        optimality_contribution_2 += value_two * value_two;
    }
    // Compute feasibility contribution to residual
    Real feasibility_contribution_1 = 0;
    Real feasibility_contribution_2 = 0;
    size_t number_inequalitites = dual_->size();
    for(size_t index = 0; index < number_inequalitites; ++index)
    {
        Real value = std::max(0., (*data_mng_->m_CurrentInequalityResiduals)[index]);
        feasibility_contribution_1 += value * value;
        value = (*dual_)[index] * std::max(0., -(*data_mng_->m_CurrentInequalityResiduals)[index]);
        feasibility_contribution_2 += value * value;
    }

    Real sum = optimality_contribution_1 + optimality_contribution_2 + feasibility_contribution_1
            + feasibility_contribution_2;
    Real norm_residual = (static_cast<Real>(1.) / static_cast<Real>(number_primals)) * std::sqrt(sum);

    return (norm_residual);
}

}

}
