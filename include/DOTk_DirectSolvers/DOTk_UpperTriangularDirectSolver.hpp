/*
 * DOTk_UpperTriangularDirectSolver.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_UPPERTRIANGULARDIRECTSOLVER_HPP_
#define DOTK_UPPERTRIANGULARDIRECTSOLVER_HPP_

#include "DOTk_DirectSolver.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class matrix;

class DOTk_UpperTriangularDirectSolver : public dotk::DOTk_DirectSolver
{
public:
    DOTk_UpperTriangularDirectSolver();
    explicit DOTk_UpperTriangularDirectSolver(size_t num_unknowns_);
    virtual ~DOTk_UpperTriangularDirectSolver();

    void backwardSolve(const std::tr1::shared_ptr<dotk::matrix<Real> > matrix_,
                       const std::vector<Real> & rhs_vec_,
                       std::vector<Real> & solution_vec_) const;

    virtual void solve(const std::tr1::shared_ptr<dotk::matrix<Real> > matrix_,
                       const std::vector<Real> & rhs_vec_,
                       std::vector<Real> & solution_vec_);

private:
    DOTk_UpperTriangularDirectSolver(const dotk::DOTk_UpperTriangularDirectSolver &);
    dotk::DOTk_UpperTriangularDirectSolver & operator=(const dotk::DOTk_UpperTriangularDirectSolver &);
};

}

#endif /* DOTK_UPPERTRIANGULARDIRECTSOLVER_HPP_ */
