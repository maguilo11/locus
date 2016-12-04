/*
 * TRROM_MxReducedBasisPDETest.cpp
 *
 *  Created on: Dec 2, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>

#include "TRROM_MxUtils.hpp"
#include "TRROM_MxVector.hpp"
#include "TRROM_MxTestUtils.hpp"
#include "TRROM_ReducedBasisPDE.hpp"
#include "TRROM_ReducedBasisData.hpp"

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
    explicit MxReducedBasisPDE(const mxArray* input_) :
            m_Solve(mxDuplicateArray(mxGetField(input_, 0, "solve"))),
            m_ApplyInverseJacobianState(mxDuplicateArray(mxGetField(input_, 0, "applyInverseJacobianState"))),
            m_ApplyAdjointInverseJacobianState(mxDuplicateArray(mxGetField(input_, 0, "applyAdjointInverseJacobianState"))),
            m_PartialDerivativeState(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeState"))),
            m_PartialDerivativeControl(mxDuplicateArray(mxGetField(input_, 0, "partialDerivativeControl"))),
            m_AdjointPartialDerivativeState(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeState"))),
            m_AdjointPartialDerivativeControl(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeControl"))),
            m_AdjointPartialDerivativeStateState(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeStateState"))),
            m_AdjointPartialDerivativeControlState(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeControlState"))),
            m_AdjointPartialDerivativeStateControl(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeStateControl"))),
            m_AdjointPartialDerivativeControlControl(mxDuplicateArray(mxGetField(input_, 0, "adjointPartialDerivativeControlControl")))
    {
    }
    //! MxReducedBasisPDE destructor.
    virtual ~MxReducedBasisPDE()
    {
        mxDestroyArray(m_AdjointPartialDerivativeControlControl);
        mxDestroyArray(m_AdjointPartialDerivativeStateControl);
        mxDestroyArray(m_AdjointPartialDerivativeControlState);
        mxDestroyArray(m_AdjointPartialDerivativeStateState);
        mxDestroyArray(m_AdjointPartialDerivativeControl);
        mxDestroyArray(m_AdjointPartialDerivativeState);
        mxDestroyArray(m_PartialDerivativeControl);
        mxDestroyArray(m_PartialDerivativeState);
        mxDestroyArray(m_ApplyAdjointInverseJacobianState);
        mxDestroyArray(m_ApplyInverseJacobianState);
        mxDestroyArray(m_Solve);
    }
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
    virtual void solve(const trrom::Vector<double> & control_,
                       trrom::Vector<double> & solution_,
                       trrom::ReducedBasisData & data_)
    {
        // Dynamic cast of active indices vector data structure
        trrom::MxVector & active_indices = dynamic_cast<trrom::MxVector &>(data_.getLeftHandSideActiveIndices());

        // Get fidelity, there are two options: low- or high-fidelity
        mxArray* mx_fidelity;
        if(data_.fidelity() == trrom::types::HIGH_FIDELITY)
        {
            mx_fidelity = mxCreateString("HIGH_FIDELITY");
            active_indices.fill(1);
        }
        else
        {
            mx_fidelity = mxCreateString("LOW_FIDELITY");
        }

        // Set control input
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());

        // Call solve function through the mex interface
        mxArray* mx_output[3];
        mxArray* mx_input[4] = { m_Solve, mx_control, active_indices.array(), mx_fidelity };
        mxArray* error = mexCallMATLABWithTrapWithObject(3, mx_output, 4, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling solve.\n";
        trrom::mx::handleException(error, msg.str());

        // Set solution output only when high fidelity solve is active
        if(data_.fidelity() == trrom::types::HIGH_FIDELITY)
        {
            assert(static_cast<size_t>(solution_.size()) == mxGetNumberOfElements(mx_output[0]));
            trrom::MxVector & solution = dynamic_cast<trrom::MxVector &>(solution_);
            solution.setMxArray(mx_output[0]);
        }

        // Set left-hand side snapshot output
        assert(static_cast<size_t>(data_.getLeftHandSideSnapshot().size()) == mxGetNumberOfElements(mx_output[1]));
        trrom::MxVector & lhs_snapshot = dynamic_cast<trrom::MxVector &>(data_.getLeftHandSideSnapshot());
        lhs_snapshot.setMxArray(mx_output[1]);

        // Set right-hand side snapshot output
        assert(static_cast<size_t>(data_.getRightHandSideSnapshot().size()) == mxGetNumberOfElements(mx_output[2]));
        trrom::MxVector & rhs_snapshot = dynamic_cast<trrom::MxVector &>(data_.getRightHandSideSnapshot());
        rhs_snapshot.setMxArray(mx_output[2]);

        mxDestroyArray(mx_fidelity);
    }
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
    virtual void applyInverseJacobianState(const trrom::Vector<double> & state_,
                                           const trrom::Vector<double> & control_,
                                           const trrom::Vector<double> & rhs_,
                                           trrom::Vector<double> & solution_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & right_hand_side = dynamic_cast<const trrom::MxVector &>(rhs_);
        mxArray* mx_right_hand_side = const_cast<mxArray*>(right_hand_side.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[4] = { m_ApplyInverseJacobianState, mx_state, mx_control, mx_right_hand_side };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling applyInverseJacobianState.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for solution field
        assert(static_cast<size_t>(solution_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & solution = dynamic_cast<trrom::MxVector &>(solution_);
        trrom::mx::setMxArray(mx_output[0], solution.array());
    }
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
    virtual void applyAdjointInverseJacobianState(const trrom::Vector<double> & state_,
                                                  const trrom::Vector<double> & control_,
                                                  const trrom::Vector<double> & rhs_,
                                                  trrom::Vector<double> & solution_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & right_hand_side = dynamic_cast<const trrom::MxVector &>(rhs_);
        mxArray* mx_right_hand_side = const_cast<mxArray*>(right_hand_side.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[4] = { m_ApplyAdjointInverseJacobianState, mx_state, mx_control, mx_right_hand_side };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling applyAdjointInverseJacobianState.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for solution field
        assert(static_cast<size_t>(solution_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & solution = dynamic_cast<trrom::MxVector &>(solution_);
        trrom::mx::setMxArray(mx_output[0], solution.array());
    }
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
    virtual void partialDerivativeState(const trrom::Vector<double> & state_,
                                        const trrom::Vector<double> & control_,
                                        const trrom::Vector<double> & vector_,
                                        trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
        mxArray* mx_vector = const_cast<mxArray*>(vector.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[4] = { m_PartialDerivativeState, mx_state, mx_control, mx_vector };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling partialDerivativeState.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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
    virtual void partialDerivativeControl(const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          const trrom::Vector<double> & vector_,
                                          trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
        mxArray* mx_vector = const_cast<mxArray*>(vector.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[4] = { m_PartialDerivativeControl, mx_state, mx_control, mx_vector };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling partialDerivativeControl.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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
    virtual void adjointPartialDerivativeState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & dual_,
                                               trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
        mxArray* mx_dual = const_cast<mxArray*>(dual.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[4] = { m_AdjointPartialDerivativeState, mx_state, mx_control, mx_dual };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling adjointPartialDerivativeState.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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
    virtual void adjointPartialDerivativeControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & dual_,
                                                 trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
        mxArray* mx_dual = const_cast<mxArray*>(dual.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[4] = { m_AdjointPartialDerivativeControl, mx_state, mx_control, mx_dual };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling adjointPartialDerivativeControl.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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
    virtual void adjointPartialDerivativeStateState(const trrom::Vector<double> & state_,
                                                    const trrom::Vector<double> & control_,
                                                    const trrom::Vector<double> & dual_,
                                                    const trrom::Vector<double> & vector_,
                                                    trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
        mxArray* mx_dual = const_cast<mxArray*>(dual.array());
        const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
        mxArray* mx_vector = const_cast<mxArray*>(vector.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[5] = { m_AdjointPartialDerivativeStateState, mx_state, mx_control, mx_dual, mx_vector };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling adjointPartialDerivativeStateState.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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
    virtual void adjointPartialDerivativeStateControl(const trrom::Vector<double> & state_,
                                                      const trrom::Vector<double> & control_,
                                                      const trrom::Vector<double> & dual_,
                                                      const trrom::Vector<double> & vector_,
                                                      trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
        mxArray* mx_dual = const_cast<mxArray*>(dual.array());
        const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
        mxArray* mx_vector = const_cast<mxArray*>(vector.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[5] = { m_AdjointPartialDerivativeStateControl, mx_state, mx_control, mx_dual, mx_vector };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling adjointPartialDerivativeStateControl.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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
    virtual void adjointPartialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                        const trrom::Vector<double> & control_,
                                                        const trrom::Vector<double> & dual_,
                                                        const trrom::Vector<double> & vector_,
                                                        trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
        mxArray* mx_dual = const_cast<mxArray*>(dual.array());
        const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
        mxArray* mx_vector = const_cast<mxArray*>(vector.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[5] = { m_AdjointPartialDerivativeControlControl, mx_state, mx_control, mx_dual, mx_vector };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling adjointPartialDerivativeControlControl.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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
    virtual void adjointPartialDerivativeControlState(const trrom::Vector<double> & state_,
                                                      const trrom::Vector<double> & control_,
                                                      const trrom::Vector<double> & dual_,
                                                      const trrom::Vector<double> & vector_,
                                                      trrom::Vector<double> & output_)
    {
        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
        mxArray* mx_state = const_cast<mxArray*>(state.array());
        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
        mxArray* mx_control = const_cast<mxArray*>(control.array());
        const trrom::MxVector & dual = dynamic_cast<const trrom::MxVector &>(dual_);
        mxArray* mx_dual = const_cast<mxArray*>(dual.array());
        const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
        mxArray* mx_vector = const_cast<mxArray*>(vector.array());

        // Call partial derivative evaluation through the mex interface
        mxArray* mx_output[1];
        mxArray* mx_input[5] = { m_AdjointPartialDerivativeControlState, mx_state, mx_control, mx_dual, mx_vector };
        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 5, mx_input, "feval");
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> Error while calling adjointPartialDerivativeControlState.\n";
        trrom::mx::handleException(error, msg.str());

        // Set output for partial derivative
        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
        trrom::mx::setMxArray(mx_output[0], output.array());
    }
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

void mexFunction(int nOutput, mxArray* pOutput[], int nInput, const mxArray* pInput[])
{
    std::string msg("\nTESTING MEX REDUCED BASIS PDE CONSTRAINT INTERFACE\n");
    mexPrintf("%s", msg.c_str());
    if(!(nInput == 1 && nOutput == 0))
    {
        std::string error("\nINCORRECT NUMBER OF INPUT AND OUTPUT AGUMENTS. FUNCTION TAKES ONE INPUT AND RETURNS NO OUTPUT.\n");
        mexErrMsgTxt(error.c_str());
    }

    // Set data for unit test
    const int num_states = 9;
    trrom::MxVector dual(num_states, 3.);
    trrom::MxVector states(num_states, 1.);
    trrom::MxVector output(num_states, 1.);
    trrom::MxVector delta_states(num_states, 2.);

    const int num_controls = 9;
    trrom::MxVector controls(num_controls, 1.);
    trrom::MxVector delta_controls(num_controls, 3.);

    trrom::ReducedBasisData data;
    const int lhs_snapshot_length = num_states * num_states;
    trrom::MxVector left_hand_side(lhs_snapshot_length);
    data.allocateLeftHandSideSnapshot(left_hand_side);
    trrom::MxVector right_hand_side(num_states, 1.);
    data.allocateRightHandSideSnapshot(right_hand_side);
    data.allocateLeftHandSideActiveIndices(left_hand_side);

    // Instantiate instance from class MxReducedBasisPDE
    trrom::MxReducedBasisPDE equality(pInput[0]);

    // **** TEST APPLY INVERSE JACOBIAN STATE ****
    msg.assign("applyInverseJacobianState");
    equality.applyInverseJacobianState(states, controls, right_hand_side, output);
    trrom::MxVector gold(num_states, 2.);
    bool did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST APPLY ADJOINT INVERSE JACOBIAN STATE ****
    msg.assign("applyAdjointInverseJacobianState");
    equality.applyAdjointInverseJacobianState(states, controls, right_hand_side, output);
    gold.fill(3);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("partialDerivativeState");
    equality.partialDerivativeState(states, controls, delta_states, output);
    gold.fill(6);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("partialDerivativeControl");
    equality.partialDerivativeControl(states, controls, delta_states, output);
    gold.fill(4);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("adjointPartialDerivativeState");
    equality.adjointPartialDerivativeState(states, controls, dual, output);
    gold.fill(5);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("adjointPartialDerivativeControl");
    equality.adjointPartialDerivativeControl(states, controls, dual, output);
    gold.fill(10);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL AND STATE VARIABLES ****
    msg.assign("adjointPartialDerivativeControlState");
    equality.adjointPartialDerivativeControlState(states, controls, dual, delta_states, output);
    gold.fill(7);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT OF SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE CONTROL VARIABLES ****
    msg.assign("adjointPartialDerivativeControlControl");
    equality.adjointPartialDerivativeControlControl(states, controls, dual, delta_controls, output);
    gold.fill(20);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT OF SECOND ORDER PARTIAL DERIVATIVE WITH RESPECT TO THE STATE VARIABLES ****
    msg.assign("adjointPartialDerivativeStateState");
    equality.adjointPartialDerivativeStateState(states, controls, dual, delta_states, output);
    gold.fill(19);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST ADJOINT MIXED PARTIAL DERIVATIVE WITH RESPECT TO THE STATE AND CONTROL VARIABLES ****
    msg.assign("adjointPartialDerivativeStateControl");
    equality.adjointPartialDerivativeStateControl(states, controls, dual, delta_controls, output);
    gold.fill(37);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    // **** TEST HIGH FIDELITY SOLVE ****
    msg.assign("solve - high fidelity - output");
    data.fidelity(trrom::types::HIGH_FIDELITY);
    equality.solve(controls, output, data);
    gold.fill(23);
    did_test_pass = trrom::mx::checkResults(gold, output);
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("solve - high fidelity - rhs");
    for(int index = 0; index < gold.size(); ++index)
    {
        gold[index] = index + 1;
    }
    did_test_pass = trrom::mx::checkResults(gold, data.getRightHandSideSnapshot());
    trrom::mx::assert_test(msg, did_test_pass);

    msg.assign("solve - high fidelity - lhs");
    trrom::MxVector lhs_gold(lhs_snapshot_length, 2.);
    for(int index = 0; index < lhs_gold.size(); ++index)
    {
        lhs_gold[index] = index + 1;
    }
    did_test_pass = trrom::mx::checkResults(lhs_gold, data.getLeftHandSideSnapshot());
    trrom::mx::assert_test(msg, did_test_pass);
}
