/*
 * DOTk_TangentialSubProblemCriterion.hpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TANGENTIALSUBPROBLEMCRITERION_HPP_
#define DOTK_TANGENTIALSUBPROBLEMCRITERION_HPP_

#include "DOTk_KrylovSolverStoppingCriterion.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<class Type>
class vector;

class DOTk_TangentialSubProblemCriterion: public dotk::DOTk_KrylovSolverStoppingCriterion
{
public:
    explicit DOTk_TangentialSubProblemCriterion(Real projected_gradient_tolerance_ = 1e-4);
    virtual ~DOTk_TangentialSubProblemCriterion();

    virtual Real evaluate(const dotk::DOTk_KrylovSolver * const solver_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_);

private:
    void initialize(Real value_);

    Real computeStoppingTolerance(Real norm_residual_,
                                  const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_);
    Real computeInitialStoppingTolerance(Real trust_region_radius_,
                                         const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_);

private:
    DOTk_TangentialSubProblemCriterion(const dotk::DOTk_TangentialSubProblemCriterion &);
    dotk::DOTk_TangentialSubProblemCriterion & operator=(const dotk::DOTk_TangentialSubProblemCriterion &);
};

}

#endif /* DOTK_TANGENTIALSUBPROBLEMCRITERION_HPP_ */
