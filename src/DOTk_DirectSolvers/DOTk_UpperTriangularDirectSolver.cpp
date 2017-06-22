/*
 * DOTk_UpperTriangularDirectSolver.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cassert>
#include "matrix.hpp"
#include "DOTk_UpperTriangularDirectSolver.hpp"

namespace dotk
{

DOTk_UpperTriangularDirectSolver::DOTk_UpperTriangularDirectSolver() :
        dotk::DOTk_DirectSolver(dotk::types::UPPER_TRIANGULAR_DIRECT_SOLVER)
{
}

DOTk_UpperTriangularDirectSolver::DOTk_UpperTriangularDirectSolver(size_t num_unknowns_) :
        dotk::DOTk_DirectSolver(dotk::types::UPPER_TRIANGULAR_DIRECT_SOLVER, num_unknowns_)
{
}

DOTk_UpperTriangularDirectSolver::~DOTk_UpperTriangularDirectSolver()
{
}

void DOTk_UpperTriangularDirectSolver::backwardSolve(const std::shared_ptr<dotk::matrix<Real> > matrix_,
                                                     const std::vector<Real> & rhs_vec_,
                                                     std::vector<Real> & solution_vec_) const
{
    size_t num_unknowns = dotk::DOTk_DirectSolver::getNumUnknowns();

    Real value = 0;
    for(int row = num_unknowns - 1; row >= 0; --row)
    {
        solution_vec_[row] = rhs_vec_[row];
        for(size_t column = row + 1; column < num_unknowns; ++column)
        {
            value = solution_vec_[column];
            solution_vec_[row] -= matrix_->operator()(row, column) * value;
        }
        solution_vec_[row] /=  matrix_->operator()(row, row);
    }
}

void DOTk_UpperTriangularDirectSolver::solve(const std::shared_ptr<dotk::matrix<Real> > matrix_,
                                             const std::vector<Real> & rhs_vec_,
                                             std::vector<Real> & solution_vec_)
{
    this->backwardSolve(matrix_, rhs_vec_, solution_vec_);
}

}
