/*
 * TRROM_TeuchosSerialDenseSolver.cpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#include "Teuchos_SerialDenseSolver.hpp"

#include "TRROM_TeuchosSerialDenseVector.hpp"
#include "TRROM_TeuchosSerialDenseMatrix.hpp"
#include "TRROM_TeuchosSerialDenseSolver.hpp"

namespace trrom
{

TeuchosSerialDenseSolver::TeuchosSerialDenseSolver() :
        m_Solver(new Teuchos::SerialDenseSolver<int, double>)
{
}

TeuchosSerialDenseSolver::~TeuchosSerialDenseSolver()
{
}

void TeuchosSerialDenseSolver::solve(const trrom::Matrix<double> & A_,
                                     const trrom::Vector<double> & rhs_,
                                     trrom::Vector<double> & lhs_)
{
    // NOTE: Matrix has to be set before the RHS and LHS vectors. If not, a segmentation fault is thrown
    int num_rows = A_.getNumRows();
    int num_columns = A_.getNumCols();
    trrom::TeuchosSerialDenseMatrix matrix(num_rows, num_columns);
    matrix.update(1., A_, 0.);
    m_Solver->setMatrix(Teuchos::rcp(&(*matrix.data()), false));
    trrom::TeuchosSerialDenseVector rhs(num_rows);
    rhs.update(1., rhs_, 0.);
    trrom::TeuchosSerialDenseVector lhs(num_columns);
    m_Solver->setVectors(Teuchos::rcp(&(*lhs.data()), false),Teuchos::rcp(&(*rhs.data()), false));

    m_Solver->factorWithEquilibration(true);
    m_Solver->factor();
    m_Solver->solve();

    lhs_.update(1., lhs, 0.);
}

}
