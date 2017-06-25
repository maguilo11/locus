/*
 * TRROM_TeuchosSerialDenseSolver.hpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#ifndef TRROM_TEUCHOSSERIALDENSESOLVER_HPP_
#define TRROM_TEUCHOSSERIALDENSESOLVER_HPP_

#include "TRROM_SolverInterface.hpp"

namespace Teuchos
{
template<typename OrdinalType, typename ScalarType>
class SerialDenseSolver;
}

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class TeuchosSerialDenseSolver : public trrom::SolverInterface
{
public:
    TeuchosSerialDenseSolver();
    virtual ~TeuchosSerialDenseSolver();

    void solve(const trrom::Matrix<double> & A_, const trrom::Vector<double> & rhs_, trrom::Vector<double> & lhs_);

private:
    std::shared_ptr< Teuchos::SerialDenseSolver<int, double> > m_Solver;

private:
    TeuchosSerialDenseSolver(const trrom::TeuchosSerialDenseSolver &);
    trrom::TeuchosSerialDenseSolver & operator=(const trrom::TeuchosSerialDenseSolver & rhs_);
};

}

#endif /* TRROM_TEUCHOSSERIALDENSESOLVER_HPP_ */
