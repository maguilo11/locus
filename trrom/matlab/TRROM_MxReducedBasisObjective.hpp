///*
// * TRROM_MxReducedBasisObjective.hpp
// *
// *  Created on: Feb 10, 2017
// *      Author: Miguel A. Aguilo Valentin
// */
//
//#ifndef TRROM_MXREDUCEDBASISOBJECTIVE_HPP_
//#define TRROM_MXREDUCEDBASISOBJECTIVE_HPP_
//
//#include <mex.h>
//#include <cassert>
//#include <sstream>
//
//#include "TRROM_Types.hpp"
//#include "TRROM_MxUtils.hpp"
//#include "TRROM_MxVector.hpp"
//
//namespace trrom
//{
//
//template<typename ScalarType>
//class Vector;
//
//class ReducedBasisObjective
//{
//public:
//    virtual ~ReducedBasisObjective()
//    {
//    }
//    ;
//
//    /*!
//     * Evaluates nonlinear programming objective function of type f(\mathbf{u},\mathbf{z})
//     * \colon\mathbb{R}^{n_u}\times\mathbb{R}^{n_z}\rightarrow\mathbb{R}, using the MEX
//     * interface. Here u denotes the state and z denotes the control variables.
//     **/
//    virtual double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_) = 0;
//    /*!
//     * Evaluates partial derivative of the objective function with respect to the
//     * control variables, \frac{\partial{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}}.
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *    \param Out
//     *          output_: partial derivative of the objective function with respect to the
//     *                   control variables
//     **/
//    virtual void gradient(const trrom::Vector<double> & state_,
//                          const trrom::Vector<double> & control_,
//                          const trrom::Vector<double> & dual_,
//                          trrom::Vector<double> & output_) = 0;
//    /*!
//     * Evaluates second partial derivative of the objective function with respect to the
//     * control variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}^2}.
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *    \param In
//     *          dual_: dual variables
//     *    \param In
//     *          vector_: direction (vector)
//     *    \param Out
//     *          output_: second partial derivative of the objective function with respect to the
//     *                   control variables
//     **/
//    virtual void hessian(const trrom::Vector<double> & state_,
//                         const trrom::Vector<double> & control_,
//                         const trrom::Vector<double> & vector_,
//                         trrom::Vector<double> & output_) = 0;
//    /*! Evaluates current objective function inexactness
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *
//     *  \return Objective function inexactness. If no error certification is available (i.e. user has
//     *  not perform the error analysis for the problem of interest), user must return zero.
//     **/
//    virtual double evaluateObjectiveInexactness(const trrom::Vector<double> & state_,
//                                                const trrom::Vector<double> & control_) = 0;
//    /*! Evaluates current gradient inexactness
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *
//     *  \return Gradient operator inexactness. If no error certification is available (i.e. user has
//     *  not perform the error analysis for the problem of interest), user must return zero.
//     **/
//    virtual double evaluateGradientInexactness(const trrom::Vector<double> & state_,
//                                               const trrom::Vector<double> & control_) = 0;
//    /*!
//     * Sets the model fidelity flag, options are low- or high-fidelity. Low- and high-fidelity model
//     * evaluations cannot be simultaneously active. Only one model evaluation mode can be active at
//     * a given instance.
//     *  Parameters:
//     *    \param In
//     *          input_: fidelity flag
//     */
//    virtual void fidelity(trrom::types::fidelity_t input_) = 0;
//};
//
//class MxReducedBasisObjective : public trrom::ReducedBasisObjective
//{
//public:
//    //! @name Constructors/destructors
//    //@{
//    /*!
//     * Creates a MxReducedBasisObjective object
//     * Parameters:
//     *    \param In
//     *          input_: MEX array pointer
//     *
//     * \return Reference to MxReducedBasisObjective.
//     *
//     **/
//    explicit MxReducedBasisObjective(const mxArray* input_) :
//            m_Value(mxDuplicateArray(mxGetField(input_, 0, "value"))),
//            m_Hessian(mxDuplicateArray(mxGetField(input_, 0, "hessian"))),
//            m_Fidelity(mxDuplicateArray(mxGetField(input_, 0, "fidelity"))),
//            m_Gradient(mxDuplicateArray(mxGetField(input_, 0, "gradient"))),
//            m_GradientError(mxDuplicateArray(mxGetField(input_, 0, "evaluateGradientInexactness"))),
//            m_ObjectiveError(mxDuplicateArray(mxGetField(input_, 0, "evaluateObjectiveInexactness")))
//    {
//    }
//    //! MxReducedBasisObjectiveOperators destructor.
//    virtual ~MxReducedBasisObjective()
//    {
//        mxDestroyArray(m_ObjectiveError);
//        mxDestroyArray(m_GradientError);
//        mxDestroyArray(m_Gradient);
//        mxDestroyArray(m_Fidelity);
//        mxDestroyArray(m_Hessian);
//        mxDestroyArray(m_Value);
//    }
//    //@}
//
//    /*!
//     * MEX interface to objective function evaluation: Evaluates nonlinear programming objective
//     * function of type f(\mathbf{u},\mathbf{z})\colon\mathbb{R}^{n_u}\times\mathbb{R}^{n_z}
//     * \rightarrow\mathbb{R}, using the MEX interface. Here u denotes the state and z denotes
//     * the control variables.
//     **/
//    double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_)
//    {
//        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
//        mxArray* mx_state = const_cast<mxArray*>(state.array());
//        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
//        mxArray* mx_control = const_cast<mxArray*>(control.array());
//
//        // Call objective function evaluation through the MEX interface
//        mxArray* mx_output[1];
//        mxArray* mx_input[3] =
//            { m_Value, mx_state, mx_control };
//        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
//        std::ostringstream msg;
//        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling value.\n";
//        trrom::mx::handleException(error, msg.str());
//
//        // Get objective function value from MATLAB's output
//        assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
//        double output = mxGetScalar(mx_output[0]);
//        return (output);
//    }
//    /*!
//     * Evaluates objective function gradient, \frac{\partial{f(\mathbf{u},\mathbf{z})}}
//     * {\partial\mathbf{z}}.
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *    \param Out
//     *          output_: objective function gradient
//     **/
//    void gradient(const trrom::Vector<double> & state_,
//                  const trrom::Vector<double> & control_,
//                  trrom::Vector<double> & output_)
//    {
//        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
//        mxArray* mx_state = const_cast<mxArray*>(state.array());
//        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
//        mxArray* mx_control = const_cast<mxArray*>(control.array());
//
//        // Call gradient through the MEX interface
//        mxArray* mx_output[1];
//        mxArray* mx_input[3] =
//            { m_Gradient, mx_state, mx_control };
//        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
//        std::ostringstream msg;
//        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling gradient.\n";
//        trrom::mx::handleException(error, msg.str());
//
//        // Set objective function gradient
//        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
//        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
//        output.setMxArray(mx_output[0]);
//    }
//    /*!
//     * Apply vector to the objective function Hessian operator, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}
//     * {\partial\mathbf{z}^2}.
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *    \param In
//     *          vector_: direction (vector)
//     *    \param Out
//     *          output_: apply vector to the Hessian operator
//     **/
//    void hessian(const trrom::Vector<double> & state_,
//                 const trrom::Vector<double> & control_,
//                 const trrom::Vector<double> & vector_,
//                 trrom::Vector<double> & output_)
//    {
//        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
//        mxArray* mx_state = const_cast<mxArray*>(state.array());
//        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
//        mxArray* mx_control = const_cast<mxArray*>(control.array());
//        const trrom::MxVector & vector = dynamic_cast<const trrom::MxVector &>(vector_);
//        mxArray* mx_vector = const_cast<mxArray*>(vector.array());
//
//        // Call Hessian through the MEX interface
//        mxArray* mx_output[1];
//        mxArray* mx_input[4] =
//            { m_Hessian, mx_state, mx_control, mx_vector };
//        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 4, mx_input, "feval");
//        std::ostringstream msg;
//        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling Hessian.\n";
//        trrom::mx::handleException(error, msg.str());
//
//        /* Set output for the application of input vector to the Hessian operator */
//        assert(static_cast<size_t>(output_.size()) == mxGetNumberOfElements(mx_output[0]));
//        trrom::MxVector & output = dynamic_cast<trrom::MxVector &>(output_);
//        output.setMxArray(mx_output[0]);
//    }
//    /*! MEX interface used to evaluate the objective function inexactness
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *
//     *  \return Objective function inexactness. If no error certification is available
//     *  (i.e. user has not perform the error analysis for the problem of interest), user
//     *  must return zero.
//     **/
//    double evaluateObjectiveInexactness(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_)
//    {
//        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
//        mxArray* mx_state = const_cast<mxArray*>(state.array());
//        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
//        mxArray* mx_control = const_cast<mxArray*>(control.array());
//
//        // Call objective function evaluation through the mex interface
//        mxArray* mx_output[1];
//        mxArray* mx_input[3] =
//            { m_ObjectiveError, mx_state, mx_control };
//        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
//        std::ostringstream msg;
//        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
//                << ", -> Error while calling evaluateObjectiveInexactness.\n";
//        trrom::mx::handleException(error, msg.str());
//
//        // Get objective function value from MATLAB's output
//        assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
//        double output = mxGetScalar(mx_output[0]);
//        return (output);
//    }
//    /*! MEX interface used to evaluate the gradient operator inexactness
//     *  Parameters:
//     *    \param In
//     *          state_: state variables
//     *    \param In
//     *          control_: control variables
//     *
//     *  \return Gradient operator inexactness. If no error certification is available
//     *  (i.e. user has not perform the error analysis for the problem of interest), user
//     *   must return zero.
//     **/
//    double evaluateGradientInexactness(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_)
//    {
//        const trrom::MxVector & state = dynamic_cast<const trrom::MxVector &>(state_);
//        mxArray* mx_state = const_cast<mxArray*>(state.array());
//        const trrom::MxVector & control = dynamic_cast<const trrom::MxVector &>(control_);
//        mxArray* mx_control = const_cast<mxArray*>(control.array());
//
//        // Call objective function evaluation through the mex interface
//        mxArray* mx_output[1];
//        mxArray* mx_input[3] =
//            { m_GradientError, mx_state, mx_control };
//        mxArray* error = mexCallMATLABWithTrapWithObject(1, mx_output, 3, mx_input, "feval");
//        std::ostringstream msg;
//        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
//                << ", -> Error while calling evaluateGradientInexactness.\n";
//        trrom::mx::handleException(error, msg.str());
//
//        // Get objective function value from MATLAB's output
//        assert(static_cast<size_t>(1) == mxGetNumberOfElements(mx_output[0]));
//        double output = mxGetScalar(mx_output[0]);
//        return (output);
//    }
//    /*!
//     * MEX interface used to set the model fidelity flag. Options are low- or high-fidelity.
//     * Low- and high-fidelity model evaluations cannot be simultaneously active. Only one
//     * model evaluation mode can be active at a given instance.
//     *  Parameters:
//     *    \param In
//     *          input_: fidelity flag
//     */
//    void fidelity(trrom::types::fidelity_t input_)
//    {
//        // Get fidelity, there are two options: low- or high-fidelity
//        mxArray* mx_fidelity;
//        if(input_ == trrom::types::HIGH_FIDELITY)
//        {
//            mx_fidelity = mxCreateString("HIGH_FIDELITY");
//        }
//        else
//        {
//            mx_fidelity = mxCreateString("LOW_FIDELITY");
//        }
//
//        // Call fidelity function through MEX interface
//        mxArray* mx_output[0];
//        mxArray* mx_input[2] =
//            { m_Fidelity, mx_fidelity };
//        mxArray* error = mexCallMATLABWithTrapWithObject(0, mx_output, 2, mx_input, "feval");
//        std::ostringstream msg;
//        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Error while calling fidelity.\n";
//        trrom::mx::handleException(error, msg.str());
//    }
//
//private:
//    mxArray* m_Value;
//    mxArray* m_Hessian;
//    mxArray* m_Fidelity;
//    mxArray* m_Gradient;
//    mxArray* m_GradientError;
//    mxArray* m_ObjectiveError;
//
//private:
//    MxReducedBasisObjective(const trrom::MxReducedBasisObjective &);
//    trrom::MxReducedBasisObjective & operator=(const trrom::MxReducedBasisObjective & rhs_);
//};
//
//class ReducedBasisAssemblyMngTypeR
//{
//public:
//    //! @name Constructors/destructors
//    //@{
//    /*!
//     * Creates a ReducedBasisAssemblyMngTypeR object
//     * Parameters:
//     *    \param In
//     *          input_: MEX array pointer
//     *
//     * \return Reference to ReducedBasisAssemblyMngTypeR.
//     *
//     **/
//    ReducedBasisAssemblyMngTypeR(const std::shared_ptr<trrom::ReducedBasisData> & data_,
//                                 const std::shared_ptr<trrom::ReducedBasisInterface> & interface_,
//                                 const std::shared_ptr<trrom::ReducedBasisObjective> & objective_,
//                                 const std::shared_ptr<trrom::ReducedBasisPDE> & pde_) :
//            m_HessianCounter(0),
//            m_GradientCounter(0),
//            m_ObjectiveCounter(0),
//            m_LowFidelitySolveCounter(0),
//            m_HighFidelitySolveCounter(0),
//            m_LowFidelityAdjointSolveCounter(0),
//            m_HighFidelityAdjointSolveCounter(0),
//            m_UseFullNewtonHessian(true),
//            m_Dual(data_->dual()->create()),
//            m_State(data_->state()->create()),
//            m_HessWorkVec(data_->state()->create()),
//            m_StateWorkVec(data_->state()->create()),
//            m_ControlWorkVec(data_->control()->create()),
//            m_PDE(pde_),
//            m_Objective(objective_),
//            m_ReducedBasisInterface(interface_)
//    {
//    }
//    //! MxReducedBasisObjectiveOperators destructor.
//    virtual ~ReducedBasisAssemblyMngTypeR()
//    {
//    }
//    //@}
//
//    /*!
//     * Get current number of objective Hessian calculations/evaluations
//     * \return: number of objective Hessian evaluations
//     */
//    int getHessianCounter() const
//    {
//        return (m_HessianCounter);
//    }
//    //! Update number of Hessian calculations/evaluations
//    void updateHessianCounter()
//    {
//        m_HessianCounter++;
//    }
//    /*!
//     * Get current number of gradient calculations/evaluations
//     * \return: number of objective gradient evaluations
//     */
//    int getGradientCounter() const
//    {
//        return (m_GradientCounter);
//    }
//    //! Update current number of gradient calculations/evaluations
//    void updateGradientCounter()
//    {
//        m_GradientCounter++;
//    }
//    /*/
//     * Get current number of objective function calculations/evaluations
//     * \return: number of objective function evaluations
//     */
//    int getObjectiveCounter() const
//    {
//        return (m_ObjectiveCounter);
//    }
//    //! Update current number of objective function calculations/evaluations
//    void updateObjectiveCounter()
//    {
//        m_ObjectiveCounter++;
//    }
//    /*!
//     * Returns number of low fidelity model evaluations
//     */
//    int getLowFidelitySolveCounter() const
//    {
//        return (m_LowFidelitySolveCounter);
//    }
//    /*!
//     * Updates number of low fidelity model evaluations
//     */
//    void updateLowFidelitySolveCounter()
//    {
//        m_LowFidelitySolveCounter++;
//    }
//    /*!
//     * Returns number of high fidelity model evaluations
//     */
//    int getHighFidelitySolveCounter() const
//    {
//        return (m_HighFidelitySolveCounter);
//    }
//    /*!
//     * Updates number of high fidelity model evaluations
//     */
//    void updateHighFidelitySolveCounter()
//    {
//        m_HighFidelitySolveCounter++;
//    }
//    /*! Update low-fidelity model: Update state, dual, and non-affine left hand
//     * side matrix orthonormal bases
//     **/
//    void updateLowFidelityModel()
//    {
//        // Update orthonormal bases (state, dual, and left hand side bases)
//        m_ReducedBasisInterface->updateOrthonormalBases();
//        // Update left hand side Discrete Empirical Interpolation Method (DEIM) active indices
//        m_ReducedBasisInterface->updateLeftHandSideDeimDataStructures();
//        // Update reduced right hand side vector
//        m_ReducedBasisInterface->updateReducedStateRightHandSide();
//    }
//    /*!
//     * Get current model fidelity: what is the current model fidelity?
//     */
//    trrom::types::fidelity_t fidelity() const
//    {
//        trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
//        return (fidelity);
//    }
//    /*!
//     * Set current model fidelity
//     */
//    void fidelity(trrom::types::fidelity_t input_)
//    {
//        m_Objective->fidelity(input_);
//        m_ReducedBasisInterface->fidelity(input_);
//    }
//    /*!
//     * Set Gauss-Newton Hessian; true: Gauss-Newton active, false: full Hessian is active
//     */
//    void useGaussNewtonHessian()
//    {
//        m_UseFullNewtonHessian = false;
//    }
//    /*!
//     * Solve high fidelity problem
//     *  Parameters:
//     *    \param In
//     *          control_: control variables
//     */
//    void solveHighFidelityProblem(const trrom::Vector<double> & control_)
//    {
//        /*! Solve for \mathbf{u}(\mathbf{z})\in\mathbb{R}^{n_u}, \mathbf{K}(\mathbf{z})\mathbf{u} = \mathbf{f}. If the parametric
//         * reduced-order model (low-fidelity model) is disabled, the user is expected to provide the current nonaffine, parameter
//         * dependent matrix snapshot by calling the respective store snapshot functionality in the ReducedBasisInterface class. If
//         * the low-fidelity model is enabled, the user is expected to provide the respective reduced snapshot for the nonaffine,
//         * parameter dependent matrices and right-hand side vectors  */
//        m_PDE->solve(control_, *m_State, *m_ReducedBasisInterface->data());
//        m_ReducedBasisInterface->storeStateSnapshot(*m_State);
//        m_ReducedBasisInterface->storeLeftHandSideSnapshot(*m_ReducedBasisInterface->data()->getLeftHandSideSnapshot());
//        this->updateHighFidelitySolveCounter();
//    }
//    /*!
//     * Solve low fidelity problem
//     *  Parameters:
//     *    \param In
//     *          control_: control variables
//     */
//    void solveLowFidelityProblem(const trrom::Vector<double> & control_)
//    {
//        /*! Compute current left hand side matrix approximation using the Active Indices computed using the Discrete Empirical
//         * Interpolation Method (DEIM). Since the low-fidelity model is enabled, the third-party application code is required
//         * to only provide the reduced left hand matrix approximation in vectorized format. Thus, the user is not required to
//         * solve the low-fidelity system of equations. The low-fidelity system of equations will be solved in-situ using the
//         * (default or custom) solver interface. */
//        m_PDE->solve(control_, *m_State, *m_ReducedBasisInterface->data());
//        m_ReducedBasisInterface->solveLowFidelityProblem(*m_State);
//        this->updateLowFidelitySolveCounter();
//    }
//    void solveHighFidelityAdjointProblem(const trrom::Vector<double> & control_)
//    {
//        // Solve adjoint system of equations, \mathbf{K}(z)\lambda = \mathbf{f}_{\lambda}, where
//        // \mathbf{f}_{\lambda} = -\frac{\partial{J}(\mathbf{u},\mathbf{z})}{\partial\mathbf{u}}
//        m_PDE->applyInverseAdjointJacobianState(*m_State, control_, *m_Dual);
//        m_ReducedBasisInterface->storeDualSnapshot(*m_Dual);
//        this->updateHighFidelityAdjointSolveCounter();
//    }
//
//    void solveLowFidelityAdjointProblem(const trrom::Vector<double> & control_)
//    {
//        // Compute reduced right hand side (RHS) vector of adjoint system of equations.
//        m_Objective->partialDerivativeState(*m_State, control_, *m_StateWorkVec);
//        m_StateWorkVec->scale(-1);
//        m_ReducedBasisInterface->solveLowFidelityAdjointProblem(*m_StateWorkVec, *m_Dual);
//        this->updateLowFidelityAdjointSolveCounter();
//    }
//    /*!
//     * Evaluate objective function
//     *  Parameters:
//     *    \param In
//     *          control_: control variables
//     *    \param In
//     *          tolerance_: objective function inexactness tolerance
//     *    \param Out
//     *          inexactness_violated_: objective function inexactness flag
//     *
//     *    \return current objective function evaluation
//     */
//    double objective(const std::shared_ptr<trrom::Vector<double> > & control_,
//                     const double & tolerance_,
//                     bool & inexactness_violated_)
//    {
//        m_State->fill(0.);
//
//        trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
//        if(fidelity == trrom::types::LOW_FIDELITY)
//        {
//            this->solveLowFidelityProblem(*control_);
//        }
//        else
//        {
//            this->solveHighFidelityProblem(*control_);
//        }
//
//        double value = m_Objective->value(*m_State, *control_);
//        this->updateObjectiveCounter();
//
//        // check objective inexactness tolerance
//        inexactness_violated_ = false;
//        double objective_error = m_Objective->evaluateObjectiveInexactness(*m_State, *control_);
//        if(objective_error > tolerance_)
//        {
//            inexactness_violated_ = true;
//        }
//        return (value);
//    }
//    /*!
//     * Evaluate objective function gradient
//     *  Parameters:
//     *    \param In
//     *          control_: control variables
//     *    \param In/Out
//     *          output_: objective function gradient
//     *    \param In
//     *          tolerance_: objective function gradient inexactness tolerance
//     *    \param Out
//     *          inexactness_violated_: objective function inexactness flag
//     */
//    void gradient(const std::shared_ptr<trrom::Vector<double> > & control_,
//                  const std::shared_ptr<trrom::Vector<double> > & output_,
//                  const double & tolerance_,
//                  bool & inexactness_violated_)
//    {
//        m_Dual->fill(0.);
//        m_StateWorkVec->fill(0.);
//
//        trrom::types::fidelity_t fidelity = m_ReducedBasisInterface->fidelity();
//        if(fidelity == trrom::types::LOW_FIDELITY)
//        {
//            this->solveLowFidelityAdjointProblem(*control_);
//        }
//        else
//        {
//            this->solveHighFidelityAdjointProblem(*control_);
//        }
//
//        // Compute gradient operator
//        m_ControlWorkVec->fill(0.);
//        m_Objective->gradient(*m_State, *control_, *m_Dual, *m_ControlWorkVec);
//        this->updateGradientCounter();
//
//        // check gradient inexactness tolerance
//        inexactness_violated_ = false;
//        double gradient_error = m_Objective->evaluateGradientInexactness(*m_State, *control_);
//        if(gradient_error > tolerance_)
//        {
//            inexactness_violated_ = true;
//        }
//    }
//    /*!
//     * Evaluate application of a vector to the objective function Hessian
//     *  Parameters:
//     *    \param In
//     *          control_: control variables
//     *    \param In
//     *          vector_: vector variables (descent direction)
//     *    \param In/Out
//     *          output_: application of a vector to the Hessian operator
//     *    \param In
//     *          tolerance_: objective function inexactness tolerance
//     *    \param Out
//     *          inexactness_violated_: objective function Hessian inexactness flag
//     */
//    void hessian(const std::shared_ptr<trrom::Vector<double> > & control_,
//                 const std::shared_ptr<trrom::Vector<double> > & vector_,
//                 const std::shared_ptr<trrom::Vector<double> > & output_,
//                 const double & tolerance_,
//                 bool & inexactness_violated_)
//    {
//    }
//
//private:
//    int m_HessianCounter;
//    int m_GradientCounter;
//    int m_ObjectiveCounter;
//    int m_LowFidelitySolveCounter;
//    int m_HighFidelitySolveCounter;
//    int m_LowFidelityAdjointSolveCounter;
//    int m_HighFidelityAdjointSolveCounter;
//
//    bool m_UseFullNewtonHessian;
//
//    std::shared_ptr<trrom::Vector<double> > m_Dual;
//    std::shared_ptr<trrom::Vector<double> > m_State;
//    std::shared_ptr<trrom::Vector<double> > m_HessWorkVec;
//    std::shared_ptr<trrom::Vector<double> > m_StateWorkVec;
//    std::shared_ptr<trrom::Vector<double> > m_ControlWorkVec;
//
//    std::shared_ptr<trrom::ReducedBasisPDE> m_PDE;
//    std::shared_ptr<trrom::ReducedBasisObjective> m_Objective;
//    std::shared_ptr<trrom::ReducedBasisInterface> m_ReducedBasisInterface;
//
//private:
//    ReducedBasisAssemblyMngTypeR(const trrom::ReducedBasisAssemblyMngTypeR &);
//    trrom::ReducedBasisAssemblyMngTypeR & operator=(const trrom::ReducedBasisAssemblyMngTypeR &);
//};
//
//}
//
//#endif /* TRROM_MXREDUCEDBASISOBJECTIVE_HPP_ */
