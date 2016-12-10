/*
 * TRROM_MxReducedBasisPDE.hpp
 *
 *  Created on: Dec 3, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MATLAB_TRROM_MXREDUCEDBASISPDE_HPP_
#define TRROM_MATLAB_TRROM_MXREDUCEDBASISPDE_HPP_

#include "TRROM_ReducedBasisPDE.hpp"

namespace trrom
{

class MxReducedBasisPDE : public trrom::ReducedBasisPDE
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxReducedBasisPDE object
     * Parameters:
     *    \param In
     *          input_: MEX array pointer
     *
     * \return Reference to MxReducedBasisPDE.
     *
     **/
    explicit MxReducedBasisPDE(const mxArray* input_);
    //! MxReducedBasisPDE destructor.
    virtual ~MxReducedBasisPDE();
    //@}

    //! @name Linear system solves
    //@{
    /*!
     * Solves system of Partial Differential Equation (PDE) using the
     * MEX interface. For instance, let g(\mathbf{u},\mathbf{z}) \equiv
     * A(\mathbf{z})\mathbf{u}-b; then, \mathbf{u}=A(\mathbf{z})^{-1}b
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
    void solve(const trrom::Vector<double> & control_,
               trrom::Vector<double> & solution_,
               trrom::ReducedBasisData & data_);
    /*!
     * Solves system of partial differential equations using the
     * MEX interface. For instance, let J(\mathbf{z})\mathbf{u}-b=0;
     * then, \mathbf{u}=J(\mathbf{z})^{-1}b. Here J denotes the
     * Jacobian operator
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
    void applyInverseJacobianState(const trrom::Vector<double> & state_,
                                   const trrom::Vector<double> & control_,
                                   const trrom::Vector<double> & rhs_,
                                   trrom::Vector<double> & solution_);
    /*!
     * Solves adjoint system of partial differential equations using the
     * MEX interface. For instance, let J^{\ast}(\mathbf{z})\mathbf{u}-b=0;
     * then, \mathbf{u}=J(\mathbf{z})^{-\ast}b. Here J denotes the Jacobian
     * operator and \ast denotes the adjoint
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
    void applyInverseAdjointJacobianState(const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          const trrom::Vector<double> & rhs_,
                                          trrom::Vector<double> & solution_);
    //@}

    //! @name First order derivatives
    //@{
    /*!
     * Evaluates partial derivative of the equality constraint with respect to the
     * state variables, \frac{\partial{g(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}},
     * using the MEX interface
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: partial derivative of the equality constraint with respect
     *          to the state variables
     **/
    void partialDerivativeState(const trrom::Vector<double> & state_,
                                const trrom::Vector<double> & control_,
                                const trrom::Vector<double> & vector_,
                                trrom::Vector<double> & output_);
    /*!
     * Evaluates partial derivative of the equality constraint with respect to the
     * control variables, \frac{\partial{g(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}},
     * using the MEX interface
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: partial derivative of the equality constraint with respect
     *          to the control variables
     **/
    void partialDerivativeControl(const trrom::Vector<double> & state_,
                                  const trrom::Vector<double> & control_,
                                  const trrom::Vector<double> & vector_,
                                  trrom::Vector<double> & output_);
    /*!
     * Evaluates adjoint partial derivative of the equality constraint with respect
     * to the state variables, \frac{\partial{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}}, using the MEX interface
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
    void adjointPartialDerivativeState(const trrom::Vector<double> & state_,
                                       const trrom::Vector<double> & control_,
                                       const trrom::Vector<double> & dual_,
                                       trrom::Vector<double> & output_);
    /*!
     * Evaluates adjoint partial derivative of the equality constraint with respect
     * to the control variables, \frac{\partial{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}}, using the MEX interface
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
    void adjointPartialDerivativeControl(const trrom::Vector<double> & state_,
                                         const trrom::Vector<double> & control_,
                                         const trrom::Vector<double> & dual_,
                                         trrom::Vector<double> & output_);
    //@}

    //! @name Second order derivatives
    //@{
    /*!
     * Evaluates adjoint of the second order partial derivative of the equality constraint
     * with respect to the state variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}^2}, using the MEX interface
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
    void adjointPartialDerivativeStateState(const trrom::Vector<double> & state_,
                                            const trrom::Vector<double> & control_,
                                            const trrom::Vector<double> & dual_,
                                            const trrom::Vector<double> & vector_,
                                            trrom::Vector<double> & output_);
    /*!
     * Evaluates adjoint of the mixed partial derivative of the equality constraint with respect
     * to the state and control variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}\partial\mathbf{z}}, using the MEX interface
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
    void adjointPartialDerivativeStateControl(const trrom::Vector<double> & state_,
                                              const trrom::Vector<double> & control_,
                                              const trrom::Vector<double> & dual_,
                                              const trrom::Vector<double> & vector_,
                                              trrom::Vector<double> & output_);
    /*!
     * Evaluates adjoint of the second order partial derivative of the equality constraint
     * with respect to the control variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}^2}, using the MEX interface
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
    void adjointPartialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                const trrom::Vector<double> & control_,
                                                const trrom::Vector<double> & dual_,
                                                const trrom::Vector<double> & vector_,
                                                trrom::Vector<double> & output_);
    /*!
     * Evaluates adjoint of the mixed partial derivative of the equality constraint with respect
     * to the control and state variables, \frac{\partial^2{g^{\ast}(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}\partial\mathbf{u}}, using the MEX interface
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
    void adjointPartialDerivativeControlState(const trrom::Vector<double> & state_,
                                              const trrom::Vector<double> & control_,
                                              const trrom::Vector<double> & dual_,
                                              const trrom::Vector<double> & vector_,
                                              trrom::Vector<double> & output_);
    //@}

private:
    //! Linear system solves member variables
    mxArray* m_Solve;
    mxArray* m_ApplyInverseJacobianState;
    mxArray* m_ApplyAdjointInverseJacobianState;

    //! First order partial derivatives member variables
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;
    mxArray* m_AdjointPartialDerivativeState;
    mxArray* m_AdjointPartialDerivativeControl;

    //! Second order partial derivatives member variables
    mxArray* m_AdjointPartialDerivativeStateState;
    mxArray* m_AdjointPartialDerivativeControlState;
    mxArray* m_AdjointPartialDerivativeStateControl;
    mxArray* m_AdjointPartialDerivativeControlControl;

private:
    MxReducedBasisPDE(const trrom::MxReducedBasisPDE &);
    trrom::MxReducedBasisPDE & operator=(const trrom::MxReducedBasisPDE & rhs_);
};

}

#endif /* TRROM_MATLAB_TRROM_MXREDUCEDBASISPDE_HPP_ */
