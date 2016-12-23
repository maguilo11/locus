/*
 * TRROM_MxReducedBasisObjective.hpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MXREDUCEDBASISOBJECTIVE_HPP_
#define TRROM_MXREDUCEDBASISOBJECTIVE_HPP_

#include <mex.h>

#include "TRROM_ReducedBasisObjective.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class MxReducedBasisObjective : public trrom::ReducedBasisObjective
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxReducedBasisObjective object
     * Parameters:
     *    \param In
     *          input_: MEX array pointer
     *
     * \return Reference to MxReducedBasisObjective.
     *
     **/
    explicit MxReducedBasisObjective(const mxArray* input_);
    //! MxReducedBasisObjective destructor.
    virtual ~MxReducedBasisObjective();
    //@}

    /*!
     * MEX interface to objective function evaluation: Evaluates nonlinear programming objective
     * function of type f(\mathbf{u},\mathbf{z})\colon\mathbb{R}^{n_u}\times\mathbb{R}^{n_z}
     * \rightarrow\mathbb{R}, using the MEX interface. Here u denotes the state and z denotes
     * the control variables.
     **/
    double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_);
    /*! MEX interface used to evaluate the objective function inexactness
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *
     *  \return Objective function inexactness. If no error certification is available
     *  (i.e. user has not perform the error analysis for the problem of interest), user
     *  must return zero.
     **/
    double evaluateObjectiveInexactness(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_);
    /*! MEX interface used to evaluate the gradient operator inexactness
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *
     *  \return Gradient operator inexactness. If no error certification is available
     *  (i.e. user has not perform the error analysis for the problem of interest), user
     *   must return zero.
     **/
    double evaluateGradientInexactness(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_);
    /*!
     * MEX interface used to evaluate partial derivative of the objective function
     * with respect to the state variables, \frac{\partial{f(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the objective function with respect to
     *          the state variables
     **/
    void partialDerivativeState(const trrom::Vector<double> & state_,
                                const trrom::Vector<double> & control_,
                                trrom::Vector<double> & output_);
    /*!
     * MEX interface used to evaluate partial derivative of the objective function
     * with respect to the state variables, \frac{\partial{f(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the objective function with respect to
     *          the control variables
     **/
    void partialDerivativeControl(const trrom::Vector<double> & state_,
                                  const trrom::Vector<double> & control_,
                                  trrom::Vector<double> & output_);
    /*!
     * MEX interface used to evaluate mixed partial derivative of the objective function
     * with respect to the control and state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}\partial\mathbf{u}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the objective function with respect
     *          to the control and state variables
     **/
    void partialDerivativeControlState(const trrom::Vector<double> & state_,
                                       const trrom::Vector<double> & control_,
                                       const trrom::Vector<double> & vector_,
                                       trrom::Vector<double> & output_);
    /*!
     * MEX interface used to evaluate second order partial derivative of the objective
     * function with respect to the control variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{z}^2}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the objective function
     *          with respect to the control variables
     **/
    void partialDerivativeControlControl(const trrom::Vector<double> & state_,
                                         const trrom::Vector<double> & control_,
                                         const trrom::Vector<double> & vector_,
                                         trrom::Vector<double> & output_);
    /*!
     * MEX interface used to evaluate second order partial derivative of the objective
     * function with respect to the state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}^2}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the objective function
     *          with respect to the state variables
     **/
    void partialDerivativeStateState(const trrom::Vector<double> & state_,
                                     const trrom::Vector<double> & control_,
                                     const trrom::Vector<double> & vector_,
                                     trrom::Vector<double> & output_);
    /*!
     * MEX interface used to evaluate mixed partial derivative of the objective function
     * with respect to the control and state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}
     * {\partial\mathbf{u}\partial\mathbf{z}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the objective function
     *          with respect to the state and control variables
     **/
    void partialDerivativeStateControl(const trrom::Vector<double> & state_,
                                       const trrom::Vector<double> & control_,
                                       const trrom::Vector<double> & vector_,
                                       trrom::Vector<double> & output_);
    /*!
     * MEX interface used to set the model fidelity flag. Options are low- or high-fidelity.
     * Low- and high-fidelity model evaluations cannot be simultaneously active. Only one
     * model evaluation mode can be active at a given instance.
     *  Parameters:
     *    \param In
     *          input_: fidelity flag
     */
    void fidelity(trrom::types::fidelity_t input_);

private:
    mxArray* m_Value;
    mxArray* m_Fidelity;
    mxArray* m_GradientError;
    mxArray* m_ObjectiveError;
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;
    mxArray* m_PartialDerivativeControlState;
    mxArray* m_PartialDerivativeControlControl;
    mxArray* m_PartialDerivativeStateState;
    mxArray* m_PartialDerivativeStateControl;

private:
    MxReducedBasisObjective(const trrom::MxReducedBasisObjective &);
    trrom::MxReducedBasisObjective & operator=(const trrom::MxReducedBasisObjective & rhs_);
};

}

#endif /* TRROM_MXREDUCEDBASISOBJECTIVE_HPP_ */
