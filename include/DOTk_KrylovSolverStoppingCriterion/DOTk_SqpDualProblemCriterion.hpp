/*
 * DOTk_SqpDualProblemCriterion.hpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SQPDUALPROBLEMCRITERION_HPP_
#define DOTK_SQPDUALPROBLEMCRITERION_HPP_

#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<typename ScalarType>
class Vector;

class DOTk_SqpDualProblemCriterion : public dotk::DOTk_KrylovSolverStoppingCriterion
{
public:
    DOTk_SqpDualProblemCriterion();
    virtual ~DOTk_SqpDualProblemCriterion();

    virtual Real evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                          const std::shared_ptr<dotk::Vector<Real> > & kernel_vector_);

    void setDualTolerance(Real tolerance_);
    Real getDualTolerance() const;
    void setDualDotGradientTolerance(Real tolerance_);
    Real getDualDotGradientTolerance() const;

private:
    void initialize();
    Real computeStoppingTolerance();

private:
    Real mStoppingTolerance;

private:
    DOTk_SqpDualProblemCriterion(const dotk::DOTk_SqpDualProblemCriterion &);
    dotk::DOTk_SqpDualProblemCriterion & operator=(const dotk::DOTk_SqpDualProblemCriterion &);
};

}

#endif /* DOTK_SQPDUALPROBLEMCRITERION_HPP_ */
