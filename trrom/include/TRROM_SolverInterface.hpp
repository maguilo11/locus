/*
 * TRROM_SolverInterface.hpp
 *
 *  Created on: Oct 17, 2016
 *      Author: maguilo
 */

#ifndef TRROM_SOLVERINTERFACE_HPP_
#define TRROM_SOLVERINTERFACE_HPP_

namespace trrom
{

class SolverInterface
{
public:
    virtual ~SolverInterface()
    {
    }
    virtual void solve(const trrom::Matrix<double> & A_,
                       const trrom::Vector<double> & rhs_,
                       trrom::Vector<double> & lhs_) = 0;
};

}

#endif /* TRROM_SOLVERINTERFACE_HPP_ */
