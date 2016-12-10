/*
 * TRROM_MxDirectSolver.hpp
 *
 *  Created on: Nov 29, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXDIRECTSOLVER_HPP_
#define TRROM_MXDIRECTSOLVER_HPP_

#include "TRROM_SolverInterface.hpp"

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class MxDirectSolver : public trrom::SolverInterface
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxDirectSolver object
     * \return Reference to MxDirectSolver.
     *
     **/
    MxDirectSolver();
    //! MxDirectSolver destructor.
    virtual ~MxDirectSolver();
    //@}

    /*! Solve linear system of equations using MATLAB's direct solver (i.e. mldivide)
     * Parameters:
     *    \param In
     *          A_: 2D MEX array (matrix)
     *    \param In
     *          rhs_ : 1D MEX array (right-hand-side vector)
     *    \param Out
     *          lhs_ : 1D MEX array (left-hand-side vector, i.e. solution vector)
     */
    void solve(const trrom::Matrix<double> & A_, const trrom::Vector<double> & rhs_, trrom::Vector<double> & lhs_);

private:
    MxDirectSolver(const trrom::MxDirectSolver &);
    trrom::MxDirectSolver & operator=(const trrom::MxDirectSolver & rhs_);
};

}

#endif /* TRROM_MXDIRECTSOLVER_HPP_ */
