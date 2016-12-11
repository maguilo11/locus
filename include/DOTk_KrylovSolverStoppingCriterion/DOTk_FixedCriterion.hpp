/*
 * DOTk_FixedCriterion.hpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_FIXEDCRITERION_HPP_
#define DOTK_FIXEDCRITERION_HPP_

#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<typename ScalarType>
class Vector;

class DOTk_FixedCriterion : public dotk::DOTk_KrylovSolverStoppingCriterion
{
public:
    explicit DOTk_FixedCriterion(Real tolerance_ = 1e-8);
    virtual ~DOTk_FixedCriterion();

    virtual Real evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                          const std::tr1::shared_ptr<dotk::Vector<Real> > & kernel_vector_);

private:
    DOTk_FixedCriterion(const dotk::DOTk_FixedCriterion &);
    dotk::DOTk_FixedCriterion & operator=(const dotk::DOTk_FixedCriterion &);
};

}

#endif /* DOTK_FIXEDCRITERION_HPP_ */
