/*
 * DOTk_DirectSolver.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_DIRECTSOLVER_HPP_
#define DOTK_DIRECTSOLVER_HPP_

#include <vector>
#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_DirectSolver
{
public:
    explicit DOTk_DirectSolver(dotk::types::direct_solver_t type_);
    DOTk_DirectSolver(dotk::types::direct_solver_t type_, size_t num_unknowns_);
    virtual ~DOTk_DirectSolver();

    void setNumUnknowns(Int num_unknowns_);
    Int getNumUnknowns() const;
    void setDirectSolverType(dotk::types::direct_solver_t type_);
    dotk::types::direct_solver_t getDirectSolverType() const;

    virtual void solve(const std::tr1::shared_ptr<dotk::matrix<Real> > matrix_,
                       const std::vector<Real> & rhs_vec_,
                       std::vector<Real> & solution_vec_);

private:
    Int mNumUnknowns;
    dotk::types::direct_solver_t mDirectSolverType;

private:
    DOTk_DirectSolver(const dotk::DOTk_DirectSolver &);
    dotk::DOTk_DirectSolver & operator=(const dotk::DOTk_DirectSolver &);
};

}

#endif /* DOTK_DIRECTSOLVER_HPP_ */
