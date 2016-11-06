/*
 * DOTk_KrylovSolverStoppingCriterion.hpp
 *
 *  Created on: Dec 10, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_KRYLOVSOLVERSTOPPINGCRITERION_HPP_
#define DOTK_KRYLOVSOLVERSTOPPINGCRITERION_HPP_

#include <map>
#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_KrylovSolver;

template<typename Type>
class vector;

class DOTk_KrylovSolverStoppingCriterion
{
public:
    explicit DOTk_KrylovSolverStoppingCriterion(dotk::types::stopping_criterion_t type_);
    virtual ~DOTk_KrylovSolverStoppingCriterion();

    dotk::types::stopping_criterion_t type() const;
    void insert(dotk::types::stopping_criterion_param_t type_, Real value_ = 0);
    void set(dotk::types::stopping_criterion_param_t type_, Real value_);
    Real get(dotk::types::stopping_criterion_param_t type_) const;

    virtual Real evaluate(const dotk::DOTk_KrylovSolver* const solver_,
                          const std::tr1::shared_ptr<dotk::vector<Real> > & kernel_vector_) = 0;

private:
    dotk::types::stopping_criterion_t m_Type;
    std::map<dotk::types::stopping_criterion_param_t, Real> m_Parameters;

private:
    DOTk_KrylovSolverStoppingCriterion(const dotk::DOTk_KrylovSolverStoppingCriterion &);
};

}

#endif /* DOTK_KRYLOVSOLVERSTOPPINGCRITERION_HPP_ */
