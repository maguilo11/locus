/*
 * TRROM_DiscreteEmpiricalInterpolation.hpp
 *
 *  Created on: Dec 8, 2016
 *      Author: maguilo
 */

#ifndef TRROM_DISCRETEEMPIRICALINTERPOLATION_HPP_
#define TRROM_DISCRETEEMPIRICALINTERPOLATION_HPP_

namespace trrom
{

class SolverInterface;
class LinearAlgebraFactory;

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

class DiscreteEmpiricalInterpolation
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a DiscreteEmpiricalInterpolation object
     * Parameters:
     *    \param In
     *          solver_: reference to trrom::SolverInterface shared pointer
     *    \param In
     *          factory_: reference to trrom::LinearAlgebraFactory shared pointer
     *
     * \return Reference to DiscreteEmpiricalInterpolation.
     *
     **/
    DiscreteEmpiricalInterpolation(const std::shared_ptr<trrom::SolverInterface> & solver_,
                                   const std::shared_ptr<trrom::LinearAlgebraFactory> & factory_);
    //! DiscreteEmpiricalInterpolation destructor
    virtual ~DiscreteEmpiricalInterpolation();
    //!@}

    /*!
     * Apply Discrete Empirical Interpolation Method (DEIM) to nonlinear operator
     *    \param In
     *          basis_: current data set
     *    \param Out
     *          binary_matrix_: binary matrix with ones in the active indices and
     *          zeros for the inactive indices
     *    \param Out
     *          active_indices_: vector of active indices, indicates the active
     *          degrees of freedom for each snapshot (i.e. column)
     **/
    void apply(const std::shared_ptr<trrom::Matrix<double> > & data_,
               const std::shared_ptr<trrom::Matrix<double> > & binary_matrix_,
               std::shared_ptr<trrom::Vector<double> > & active_indices_);

private:
    std::shared_ptr<trrom::SolverInterface> m_Solver;
    std::shared_ptr<trrom::LinearAlgebraFactory> m_Factory;

private:
    DiscreteEmpiricalInterpolation(const trrom::DiscreteEmpiricalInterpolation &);
    trrom::DiscreteEmpiricalInterpolation & operator=(const trrom::DiscreteEmpiricalInterpolation &);
};

}

#endif /* TRROM_DISCRETEEMPIRICALINTERPOLATION_HPP_ */
