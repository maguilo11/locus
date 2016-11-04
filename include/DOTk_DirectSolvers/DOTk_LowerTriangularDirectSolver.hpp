/*
 * DOTk_LowerTriangularDirectSolver.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_LOWERTRIANGULARDIRECTSOLVER_HPP_
#define DOTK_LOWERTRIANGULARDIRECTSOLVER_HPP_

#include "DOTk_DirectSolver.hpp"

namespace dotk
{

template<class Type>
class vector;
template<class Type>
class matrix;

class DOTk_LowerTriangularDirectSolver : public dotk::DOTk_DirectSolver
{
public:
    DOTk_LowerTriangularDirectSolver();
    explicit DOTk_LowerTriangularDirectSolver(size_t num_unknowns_);
    virtual ~DOTk_LowerTriangularDirectSolver();

    void forwardSolve(const std::vector<std::vector<Real> > matrix_,
                      const std::vector<Real> & rhs_vec_,
                      std::vector<Real> & solution_vec_) const;

    virtual void solve(const std::vector<std::vector<Real> > matrix_,
                       const std::vector<Real> & rhs_vec_,
                       std::vector<Real> & solution_vec_);

private:
    DOTk_LowerTriangularDirectSolver(const dotk::DOTk_LowerTriangularDirectSolver &);
    dotk::DOTk_LowerTriangularDirectSolver & operator=(const dotk::DOTk_LowerTriangularDirectSolver &);
};

}

#endif /* DOTK_LOWERTRIANGULARDIRECTSOLVER_HPP_ */
