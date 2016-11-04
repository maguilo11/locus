/*
 * DOTk_LowerTriangularDirectSolver.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_LowerTriangularDirectSolver.hpp"

namespace dotk
{

DOTk_LowerTriangularDirectSolver::DOTk_LowerTriangularDirectSolver() :
        dotk::DOTk_DirectSolver::DOTk_DirectSolver(dotk::types::LOWER_TRIANGULAR_DIRECT_SOLVER)
{
}

DOTk_LowerTriangularDirectSolver::DOTk_LowerTriangularDirectSolver(size_t num_unknowns_) :
        dotk::DOTk_DirectSolver::DOTk_DirectSolver(dotk::types::LOWER_TRIANGULAR_DIRECT_SOLVER, num_unknowns_)
{
}

DOTk_LowerTriangularDirectSolver::~DOTk_LowerTriangularDirectSolver()
{
}

void DOTk_LowerTriangularDirectSolver::forwardSolve(const std::vector<std::vector<Real> > matrix_,
                                                    const std::vector<Real> & rhs_vec_,
                                                    std::vector<Real> & solution_vec_) const
{
    size_t num_unknowns = dotk::DOTk_DirectSolver::getNumUnknowns();
    for(size_t row = 0; row < num_unknowns; ++row)
    {
        solution_vec_[row] = rhs_vec_[row];
        for(size_t column = 0; column < row; ++column)
        {
            solution_vec_[row] -= matrix_[row][column] * solution_vec_[column];
        }
        size_t last_entry = matrix_[row].size() - 1;
        solution_vec_[row] /= matrix_[row][last_entry];
    }
}

void DOTk_LowerTriangularDirectSolver::solve(const std::vector<std::vector<Real> > matrix_,
                                             const std::vector<Real> & rhs_vec_,
                                             std::vector<Real> & solution_vec_)
{
    this->forwardSolve(matrix_, rhs_vec_, solution_vec_);
}

}
