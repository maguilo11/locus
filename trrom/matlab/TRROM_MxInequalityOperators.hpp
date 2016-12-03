/*
 * TRROM_MxInequalityOperators.hpp
 *
 *  Created on: Dec 2, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MXINEQUALITYOPERATORS_HPP_
#define TRROM_MXINEQUALITYOPERATORS_HPP_

#include "TRROM_InequalityOperators.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class MxInequalityOperators : public trrom::InequalityOperators
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxInequalityOperators object
     * Parameters:
     *    \param In
     *          input_: MEX array pointer
     *
     * \return Reference to MxInequalityOperators.
     *
     **/
    explicit MxInequalityOperators(const mxArray* input_);
    //! MxInequalityOperators destructor.
    virtual ~MxInequalityOperators();
    //@}

    /*!
     * MEX interface for the inequality constraint, i.e h(\mathbf{u}(\mathbf{z}),\mathbf{z})
     * \equiv value(\mathbf{u}(\mathbf{z}),\mathbf{z}) - bound \leq 0, where bound denotes
     * the condition that must be met for a given inequality constraint.
     **/
    double bound();
    /*!
     * Evaluates the current value of the user-defined inequality constraint using the
     * MEX interface. Here \mathbf{u} denotes the state and \mathbf{z} denotes the control variables.
     **/
    double value(const trrom::Vector<double> & state_, const trrom::Vector<double> & control_);
    /*!
     * Evaluates partial derivative of the inequality constraint with respect to the
     * state variables, \frac{\partial{h(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the inequality constraint with respect to
     *          the state variables
     **/
    void partialDerivativeState(const trrom::Vector<double> & state_,
                                const trrom::Vector<double> & control_,
                                trrom::Vector<double> & output_);
    /*!
     * Evaluates partial derivative of the inequality constraint with respect to the state variables,
     * \frac{\partial{h(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}}, using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param Out
     *          output_: partial derivative of the inequality constraint with respect to the control variables
     **/
    void partialDerivativeControl(const trrom::Vector<double> & state_,
                                  const trrom::Vector<double> & control_,
                                  trrom::Vector<double> & output_);
    /*!
     * Evaluates mixed partial derivative of the inequality constraint with respect to the control and
     * state variables, \frac{\partial^2{h(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}\partial\mathbf{u}},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the inequality constraint with respect to the control and state variables
     **/
    void partialDerivativeControlState(const trrom::Vector<double> & state_,
                                       const trrom::Vector<double> & control_,
                                       const trrom::Vector<double> & vector_,
                                       trrom::Vector<double> & output_);
    /*!
     * Evaluates second order partial derivative of the inequality constraint with respect to the
     * control variables, \frac{\partial^2{h(\mathbf{u},\mathbf{z})}}{\partial\mathbf{z}^2},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the inequality constraint with respect to the control variables
     **/
    void partialDerivativeControlControl(const trrom::Vector<double> & state_,
                                         const trrom::Vector<double> & control_,
                                         const trrom::Vector<double> & vector_,
                                         trrom::Vector<double> & output_);
    /*!
     * Evaluates second order partial derivative of the inequality constraint with respect to the
     * state variables, \frac{\partial^2{h(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}^2},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: second order partial derivative of the inequality constraint with respect to the state variables
     **/
    void partialDerivativeStateState(const trrom::Vector<double> & state_,
                                     const trrom::Vector<double> & control_,
                                     const trrom::Vector<double> & vector_,
                                     trrom::Vector<double> & output_);
    /*!
     * Evaluates mixed partial derivative of the inequality constraint with respect to the control and
     * state variables, \frac{\partial^2{h(\mathbf{u},\mathbf{z})}}{\partial\mathbf{u}\partial\mathbf{z}},
     * using the MEX interface.
     *  Parameters:
     *    \param In
     *          state_: state variables
     *    \param In
     *          control_: control variables
     *    \param In
     *          vector_: perturbation vector
     *    \param Out
     *          output_: mixed partial derivative of the inequality constraint with respect to the state and control variables
     **/
    void partialDerivativeStateControl(const trrom::Vector<double> & state_,
                                       const trrom::Vector<double> & control_,
                                       const trrom::Vector<double> & vector_,
                                       trrom::Vector<double> & output_);

private:
    mxArray* m_Bound;
    mxArray* m_Value;
    mxArray* m_PartialDerivativeState;
    mxArray* m_PartialDerivativeControl;
    mxArray* m_PartialDerivativeControlState;
    mxArray* m_PartialDerivativeControlControl;
    mxArray* m_PartialDerivativeStateState;
    mxArray* m_PartialDerivativeStateControl;

private:
    MxInequalityOperators(const trrom::MxInequalityOperators &);
    trrom::MxInequalityOperators & operator=(const trrom::MxInequalityOperators & rhs_);
};

}

#endif /* TRROM_MXINEQUALITYOPERATORS_HPP_ */
