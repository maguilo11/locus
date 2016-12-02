/*
 * TRROM_MxReducedObjectiveOperators.hpp
 *
 *  Created on: Dec 1, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXREDUCEDOBJECTIVEOPERATORS_HPP_
#define TRROM_MXREDUCEDOBJECTIVEOPERATORS_HPP_

#include <mex.h>
#include "TRROM_ReducedObjectiveOperators.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class MxReducedObjectiveOperators : public trrom::ReducedObjectiveOperators
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxReducedObjectiveOperators object
     * Parameters:
     *    \param In
     *          input_: MEX array pointer
     *
     * \return Reference to MxReducedObjectiveOperators.
     *
     **/
    explicit MxReducedObjectiveOperators(const mxArray* input_);
    //! MxReducedObjectiveOperators destructor.
    virtual ~MxReducedObjectiveOperators();
    //@}

    /*!
     * Evaluates nonlinear programming objective function of type f(\mathbf{u},\mathbf{z})\colon
     * \mathbb{R}^{n_u}\times\mathbb{R}^{n_z}\rightarrow\mathbb{R}, using the MEX interface. Here
     * u denotes the state and z denotes the control variables.
     **/
    double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_);
    /*! MEX interface is used to evaluate the objective function inexactness
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *
     *  \return Objective function inexactness. If no error certification is available (i.e. user has not
     *  perform the error analysis for the problem of interest), user must return zero.
     **/
    double evaluateObjectiveInexactness(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_);
    /*! MEX interface is used to evaluate the gradient operator inexactness
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *
     *  \return Gradient operator inexactness. If no error certification is available (i.e. user has not
     *  perform the error analysis for the problem of interest), user must return zero.
     **/
    double evaluateGradientInexactness(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_);
    /*!
     * Evaluates partial derivative of the objective function with respect to the
     * state variables, \frac{\partial{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the objective function with respect to the state variables
     **/
    void partialDerivativeState(const trrom::Vector<double> & state_,
                                const trrom::Vector<double> & control_,
                                trrom::Vector<double> & output_);
    /*!
     * Evaluates partial derivative of the objective function with respect to the state variables,
     * \frac{\partial{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}}, using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the objective function with respect to the control variables
     **/
    void partialDerivativeControl(const trrom::Vector<double> & state_,
                                  const trrom::Vector<double> & control_,
                                  trrom::Vector<double> & output_);
    /*!
     * Evaluates mixed partial derivative of the objective function with respect to the control and
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}\partial\mathbf{u}},
     * using the MEX interface.
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
    void partialDerivativeControlState(const trrom::Vector<double> & state_,
                                       const trrom::Vector<double> & control_,
                                       const trrom::Vector<double> & vector_,
                                       trrom::Vector<double> & output_);
    /*!
     * Evaluates second order partial derivative of the objective function with respect to the
     * control variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}^2},
     * using the MEX interface.
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
    void partialDerivativeControlControl(const trrom::Vector<double> & state_,
                                         const trrom::Vector<double> & control_,
                                         const trrom::Vector<double> & vector_,
                                         trrom::Vector<double> & output_);
    /*!
     * Evaluates second order partial derivative of the objective function with respect to the
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}^2},
     * using the MEX interface.
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
    void partialDerivativeStateState(const trrom::Vector<double> & state_,
                                     const trrom::Vector<double> & control_,
                                     const trrom::Vector<double> & vector_,
                                     trrom::Vector<double> & output_);
    /*!
     * Evaluates mixed partial derivative of the objective function with respect to the control and
     * state variables, \frac{\partial^2{f(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}\partial\mathbf{z}},
     * using the MEX interface.
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
    void partialDerivativeStateControl(const trrom::Vector<double> & state_,
                                       const trrom::Vector<double> & control_,
                                       const trrom::Vector<double> & vector_,
                                       trrom::Vector<double> & output_);

private:

    mxArray* m_Value;
    mxArray* m_GradientError;
    mxArray* m_ObjectiveError;
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;
    mxArray* m_PartialDerivativeControlState;
    mxArray* m_PartialDerivativeControlControl;
    mxArray* m_PartialDerivativeStateState;
    mxArray* m_PartialDerivativeStateControl;

private:
    MxReducedObjectiveOperators(const trrom::MxReducedObjectiveOperators &);
    trrom::MxReducedObjectiveOperators & operator=(const trrom::MxReducedObjectiveOperators & rhs_);
};

}

#endif /* TRROM_MXREDUCEDOBJECTIVEOPERATORS_HPP_ */
