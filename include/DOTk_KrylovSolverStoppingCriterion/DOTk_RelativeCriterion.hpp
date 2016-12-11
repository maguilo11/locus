/*
 * DOTk_RelativeCriterion.hpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_RELATIVECRITERION_HPP_
#define DOTK_RELATIVECRITERION_HPP_

#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<typename ScalarType>
class Vector;

class DOTk_RelativeCriterion : public dotk::DOTk_KrylovSolverStoppingCriterion
{
public:
    explicit DOTk_RelativeCriterion(Real relative_tolerance_);
    virtual ~DOTk_RelativeCriterion();

    virtual Real evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_);

private:
    Real computeStoppingTolerance(Real intial_norm_residual_);

private:
    Real m_StoppingTolerance;

private:
    DOTk_RelativeCriterion(const dotk::DOTk_RelativeCriterion &);
    dotk::DOTk_RelativeCriterion & operator=(const dotk::DOTk_RelativeCriterion &);
};

}

#endif /* DOTK_RELATIVECRITERION_HPP_ */
