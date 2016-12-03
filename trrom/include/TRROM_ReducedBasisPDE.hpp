/*
 * TRROM_ReducedBasisPDE.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_REDUCEDBASISPDE_HPP_
#define TRROM_REDUCEDBASISPDE_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class ReducedBasisData;

class ReducedBasisPDE
{
public:
    virtual ~ReducedBasisPDE()
    {
    }

    //! @name Linear system solves
    //@{
    /*!
     * Solves system of Partial Differential Equation (PDE). For instance,
     * let g(\mathbf{u},\mathbf{z}) \equiv A(\mathbf{z})\mathbf{u}-b; then,
     * \mathbf{u}=A(\mathbf{z})^{-1}b
     * Parameters:
     *    \param In
     *          control: control variables
     *    \param Out
     *          solution_: solution vector
     *    \param In/Out
     *          data_: reduced basis data structure used to access data
     *          during low fidelity PDE solves as well as set data used
     *          to generate the reduced-order model during optimization
     **/
    virtual void solve(const trrom::Vector<double> & control_,
                       trrom::Vector<double> & solution_,
                       trrom::ReducedBasisData & data_) = 0;
    /*!
     * Solves system of partial differential equations. For instance,
     * let J(\mathbf{z})\mathbf{u}-b=0; then, \mathbf{u}=J(\mathbf{z})^{-1}b.
     * Here J denotes the Jacobian operator
     * Parameters:
     *    \param In
     *          state_: state variables, e.g. displacement field for linear statics
     *    \param In
     *          control_: control_ variables
     *    \param In
     *          rhs_: right hand side vector
     *    \param Out
     *          solution_: solution vector
     **/
    virtual void applyInverseJacobianState(const trrom::Vector<double> & state_,
                                           const trrom::Vector<double> & control_,
                                           const trrom::Vector<double> & rhs_,
                                           trrom::Vector<double> & solution_) = 0;
    /*!
     * Solves adjoint system of partial differential equations. For instance,
     * let J^{\ast}(\mathbf{z})\mathbf{u}-b=0; then, \mathbf{u}=J(\mathbf{z})^{-\ast}b.
     * Here J denotes the Jacobian operator and \ast denotes the adjoint
     * Parameters:
     *    \param In
     *          state_: state variables, e.g. displacement field for linear statics
     *    \param In
     *          control_: control_ variables
     *    \param In
     *          rhs_: right hand side vector
     *    \param Out
     *          solution_: solution vector
     **/
    virtual void applyAdjointInverseJacobianState(const trrom::Vector<double> & state_,
                                                  const trrom::Vector<double> & control_,
                                                  const trrom::Vector<double> & rhs_,
                                                  trrom::Vector<double> & solution_) = 0;
    //@}

    //! @name First order derivatives
    //@{
    /*!
     * Evaluates partial derivative of the equality constraint with respect to the
     * state variables, \frac{\partial{g(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the equality constraint with respect
     *          to the state variables
     **/
    virtual void partialDerivativeState(const trrom::Vector<double> & state_,
                                        const trrom::Vector<double> & control_,
                                        const trrom::Vector<double> & vector_,
                                        trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates partial derivative of the equality constraint with respect to the
     * control variables, \frac{\partial{g(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the equality constraint with respect
     *          to the control variables
     **/
    virtual void partialDerivativeControl(const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          const trrom::Vector<double> & vector_,
                                          trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates adjoint partial derivative of the equality constraint with respect to
     * the state variables, \frac{\partial{g^{\ast}(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          dual_: dual variables, i.e. Lagrange multipliers
     *    \param Out
     *          output_: adjoint of the partial derivative of the equality constraint with
     *          respect to the state variables
     **/
    virtual void adjointPartialDerivativeState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & dual_,
                                               trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates adjoint partial derivative of the equality constraint with respect to
     * the control variables, \frac{\partial{g^{\ast}(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          dual_: dual variables, i.e. Lagrange multipliers
     *    \param Out
     *          output_: adjoint of the partial derivative of the equality constraint with
     *          respect to the control variables
     **/
    virtual void adjointPartialDerivativeControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & dual_,
                                                 trrom::Vector<double> & output_) = 0;
    //@}

    //! @name Second order derivatives
    //@{
    /*!
     * Evaluates adjoint of the second order partial derivative of the equality constraint
     * with respect to the state variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}^2}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          dual_: dual variables, i.e. Lagrange multipliers
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: adjoint of the second order partial derivative of the equality
     *          constraint with respect to the state variables
     **/
    virtual void adjointPartialDerivativeStateState(const trrom::Vector<double> & state_,
                                                    const trrom::Vector<double> & control_,
                                                    const trrom::Vector<double> & dual_,
                                                    const trrom::Vector<double> & vector_,
                                                    trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates adjoint of the mixed partial derivative of the equality constraint with respect
     * to the state and control variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}\partial\mathbf{z}}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          dual_: dual variables, i.e. Lagrange multipliers
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: adjoint of the mixed partial derivative of the equality constraint with
     *          respect to the state and control variables
     **/
    virtual void adjointPartialDerivativeStateControl(const trrom::Vector<double> & state_,
                                                      const trrom::Vector<double> & control_,
                                                      const trrom::Vector<double> & dual_,
                                                      const trrom::Vector<double> & vector_,
                                                      trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates adjoint of the second order partial derivative of the equality constraint
     * with respect to the control variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}^2}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          dual_: dual variables, i.e. Lagrange multipliers
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: adjoint of the second order partial derivative of the equality
     *          constraint with respect to the control variables
     **/
    virtual void adjointPartialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                        const trrom::Vector<double> & control_,
                                                        const trrom::Vector<double> & dual_,
                                                        const trrom::Vector<double> & vector_,
                                                        trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates adjoint of the mixed partial derivative of the equality constraint with respect
     * to the control and state variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}\partial\mathbf{u}}
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          dual_: dual variables, i.e. Lagrange multipliers
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: adjoint of the mixed partial derivative of the equality constraint with
     *          respect to the control and state variables
     **/
    virtual void adjointPartialDerivativeControlState(const trrom::Vector<double> & state_,
                                                      const trrom::Vector<double> & control_,
                                                      const trrom::Vector<double> & dual_,
                                                      const trrom::Vector<double> & vector_,
                                                      trrom::Vector<double> & output_) = 0;
    //@}
};

}

#endif /* TRROM_REDUCEDBASISPDE_HPP_ */
