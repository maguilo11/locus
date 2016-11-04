/*
 * DOTk_DirectSolver.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "matrix.hpp"
#include "DOTk_DirectSolver.hpp"

namespace dotk
{

DOTk_DirectSolver::DOTk_DirectSolver(dotk::types::direct_solver_t type_) :
        mNumUnknowns(0),
        mDirectSolverType(type_)
{
}

DOTk_DirectSolver::DOTk_DirectSolver(dotk::types::direct_solver_t type_, size_t num_unknowns_) :
        mNumUnknowns(num_unknowns_),
        mDirectSolverType(type_)
{
}

DOTk_DirectSolver::~DOTk_DirectSolver()
{
}

void DOTk_DirectSolver::setNumUnknowns(Int num_unknowns_)
{
    mNumUnknowns = num_unknowns_;
}

Int DOTk_DirectSolver::getNumUnknowns() const
{
    return (mNumUnknowns);
}

void DOTk_DirectSolver::setDirectSolverType(dotk::types::direct_solver_t type_)
{
    mDirectSolverType = type_;
}

dotk::types::direct_solver_t DOTk_DirectSolver::getDirectSolverType() const
{
    return (mDirectSolverType);
}

void DOTk_DirectSolver::solve(const std::tr1::shared_ptr<dotk::matrix<Real> > matrix_,
                              const std::vector<Real> & rhs_vec_,
                              std::vector<Real> & solution_vec_)
{
    std::string msg(" CALLING UNIMPLEMENTED dotk::DOTk_DirectSolver::solve **** ");
    std::cerr << " **** ERROR MESSAGE: " << msg.c_str() << std::flush;
    abort();
}

}
