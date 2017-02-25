/*
 * ReducedBasisObjectiveOperators.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_REDUCEDBASISOBJECTIVEOPERATORS_HPP_
#define TRROM_REDUCEDBASISOBJECTIVEOPERATORS_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class ReducedBasisObjectiveOperators
{
public:
    virtual ~ReducedBasisObjectiveOperators()
    {
    }

    /*!
     * Evaluates nonlinear programming objective function of type f(\mathbf{u},\mathbf{z})\colon
     * \mathbb{R}^{n_u}\times\mathbb{R}^{n_z}\rightarrow\mathbb{R}, using the MEX interface. Here
     * u denotes the state and z denotes the control variables.
     **/
    virtual double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_) = 0;
    /*! Evaluates current objective function inexactness
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *
     *  \return Objective function inexactness. If no error certification is available (i.e. user has not
     *  perform the error analysis for the problem of interest), user must return zero.
     **/
    virtual double evaluateObjectiveInexactness(const trrom::Vector<double> & state_,
                                                const trrom::Vector<double> & control_) = 0;
    /*! Evaluates current gradient inexactness
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *
     *  \return Gradient operator inexactness. If no error certification is available (i.e. user has not
     *  perform the error analysis for the problem of interest), user must return zero.
     **/
    virtual double evaluateGradientInexactness(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_) = 0;
    /*!
     * Evaluates partial derivative of the objective function with respect to the
     * state variables, \frac{\partial{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the objective function with respect to the state variables
     **/
    virtual void partialDerivativeState(const trrom::Vector<double> & state_,
                                        const trrom::Vector<double> & control_,
                                        trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates partial derivative of the objective function with respect to the state variables,
     * \frac{\partial{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the objective function with respect to the control variables
     **/
    virtual void partialDerivativeControl(const trrom::Vector<double> & state_,
                                          const trrom::Vector<double> & control_,
                                          trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates mixed partial derivative of the objective function with respect to the control and
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}\partial\mathbf{u}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the objective function with respect to the control and state variables
     **/
    virtual void partialDerivativeControlState(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates second order partial derivative of the objective function with respect to the
     * control variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}^2}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the objective function with respect to the control variables
     **/
    virtual void partialDerivativeControlControl(const trrom::Vector<double> & state_,
                                                 const trrom::Vector<double> & control_,
                                                 const trrom::Vector<double> & vector_,
                                                 trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates second order partial derivative of the objective function with respect to the
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}^2}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the objective function with respect to the state variables
     **/
    virtual void partialDerivativeStateState(const trrom::Vector<double> & state_,
                                             const trrom::Vector<double> & control_,
                                             const trrom::Vector<double> & vector_,
                                             trrom::Vector<double> & output_) = 0;
    /*!
     * Evaluates mixed partial derivative of the objective function with respect to the control and
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}\partial\mathbf{z}}.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the objective function with respect to the state and control variables
     **/
    virtual void partialDerivativeStateControl(const trrom::Vector<double> & state_,
                                               const trrom::Vector<double> & control_,
                                               const trrom::Vector<double> & vector_,
                                               trrom::Vector<double> & output_) = 0;
    /*!
     * Sets the model fidelity flag, options are low- or high-fidelity. Low- and high-fidelity model evaluations cannot
     * be simultaneously active. Only one model evaluation mode can be active at a given instance.
     *  Parameters:
     *    \param In
     *          input_: fidelity flag
     */
    virtual void fidelity(trrom::types::fidelity_t input_) = 0;
};

}

#endif /* TRROM_REDUCEDBASISOBJECTIVEOPERATORS_HPP_ */
