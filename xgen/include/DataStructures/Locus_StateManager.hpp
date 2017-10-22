/*
 * Locus_StateManager.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_STATEMANAGER_HPP_
#define LOCUS_STATEMANAGER_HPP_

namespace locus
{

template<typename ScalarType, typename OrdinalType>
class MultiVector;

template<typename ScalarType, typename OrdinalType = size_t>
class StateManager
{
public:
    virtual ~StateManager()
    {
    }

    virtual ScalarType evaluateObjective(const locus::MultiVector<ScalarType, OrdinalType> & aControl) = 0;
    virtual void computeGradient(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                 locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;
    virtual void applyVectorToHessian(const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                                      const locus::MultiVector<ScalarType, OrdinalType> & aVector,
                                      locus::MultiVector<ScalarType, OrdinalType> & aOutput) = 0;

    virtual ScalarType getCurrentObjectiveValue() const = 0;
    virtual void setCurrentObjectiveValue(const ScalarType & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getTrialStep() const = 0;
    virtual void setTrialStep(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentControl() const = 0;
    virtual void setCurrentControl(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getCurrentGradient() const = 0;
    virtual void setCurrentGradient(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlLowerBounds() const = 0;
    virtual void setControlLowerBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
    virtual const locus::MultiVector<ScalarType, OrdinalType> & getControlUpperBounds() const = 0;
    virtual void setControlUpperBounds(const locus::MultiVector<ScalarType, OrdinalType> & aInput) = 0;
};

} // namespace locus

#endif /* LOCUS_STATEMANAGER_HPP_ */
