/*
 * Locus_QuadraticLineSearch.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_QUADRATICLINESEARCH_HPP_
#define LOCUS_QUADRATICLINESEARCH_HPP_

#include <cmath>
#include <vector>
#include <memory>

#include "Locus_LineSearch.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_StateManager.hpp"
#include "Locus_LinearAlgebra.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class QuadraticLineSearch : public locus::LineSearch<ScalarType, OrdinalType>
{
public:
    explicit QuadraticLineSearch(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mStepLowerBound(1e-3),
            mStepUpperBound(0.5),
            mContractionFactor(0.5),
            mInitialTrialStepDotCurrentGradient(0),
            mStepValues(2, static_cast<ScalarType>(0)),
            mTrialControl(aDataFactory.control().create())
    {
    }
    virtual ~QuadraticLineSearch()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    ScalarType getStepValue() const
    {
        return (mStepValues[1]);
    }
    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setStepLowerBound(const ScalarType & aInput)
    {
        mStepLowerBound = aInput;
    }
    void setStepUpperBound(const ScalarType & aInput)
    {
        mStepUpperBound = aInput;
    }
    void setContractionFactor(const ScalarType & aInput)
    {
        mContractionFactor = aInput;
    }

    void step(locus::StateManager<ScalarType, OrdinalType> & aStateMng)
    {
        OrdinalType tSize = 3;
        // tObjectiveFunction[0] = current trial value
        // tObjectiveFunction[1] = old trial value
        // tObjectiveFunction[2] = new trial value
        std::vector<ScalarType> tObjectiveFunction(tSize);
        tObjectiveFunction[0] = aStateMng.getCurrentObjectiveValue();

        ScalarType tNormTrialStep = locus::dot(aStateMng.getTrialStep(), aStateMng.getTrialStep());
        tNormTrialStep = std::sqrt(tNormTrialStep);
        mStepValues[1] = std::min(static_cast<ScalarType>(1),
                                  static_cast<ScalarType>(100) / (static_cast<ScalarType>(1) + tNormTrialStep));

        locus::update(static_cast<ScalarType>(1),
                      aStateMng.getCurrentControl(),
                      static_cast<ScalarType>(0),
                      *mTrialControl);
        locus::update(mStepValues[1], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

        tObjectiveFunction[2] = aStateMng.evaluateObjective(*mTrialControl);
        mInitialTrialStepDotCurrentGradient = locus::dot(aStateMng.getTrialStep(), aStateMng.getCurrentGradient());
        const ScalarType tAlpha = 1e-4;
        ScalarType tTargetObjectiveValue = tObjectiveFunction[0]
                - (tAlpha * mStepValues[1] * mInitialTrialStepDotCurrentGradient);

        mNumIterationsDone = 1;
        while(tObjectiveFunction[2] > tTargetObjectiveValue)
        {
            mStepValues[0] = mStepValues[1];
            ScalarType tStep = this->interpolate(tObjectiveFunction, mStepValues);
            mStepValues[1] = tStep;

            locus::update(static_cast<ScalarType>(1),
                          aStateMng.getCurrentControl(),
                          static_cast<ScalarType>(0),
                          *mTrialControl);
            locus::update(mStepValues[1], aStateMng.getTrialStep(), static_cast<ScalarType>(1), *mTrialControl);

            tObjectiveFunction[1] = tObjectiveFunction[2];
            tObjectiveFunction[2] = aStateMng.evaluateObjective(*mTrialControl);

            mNumIterationsDone++;
            if(mNumIterationsDone >= mMaxNumIterations)
            {
                break;
            }
            tTargetObjectiveValue = tObjectiveFunction[0]
                    - (tAlpha * mStepValues[1] * mInitialTrialStepDotCurrentGradient);
        }

        aStateMng.setCurrentObjectiveValue(tObjectiveFunction[2]);
        aStateMng.setCurrentControl(*mTrialControl);
    }

private:
    ScalarType interpolate(const std::vector<ScalarType> & aObjectiveFunction, const std::vector<ScalarType> & aStepValues)
    {
        ScalarType tStepLowerBound = aStepValues[1] * mStepLowerBound;
        ScalarType tStepUpperBound = aStepValues[1] * mStepUpperBound;
        ScalarType tDenominator = static_cast<ScalarType>(2) * aStepValues[1]
                * (aObjectiveFunction[2] - aObjectiveFunction[0] - mInitialTrialStepDotCurrentGradient);

        ScalarType tStep = -mInitialTrialStepDotCurrentGradient / tDenominator;
        if(tStep < tStepLowerBound)
        {
            tStep = tStepLowerBound;
        }
        if(tStep > tStepUpperBound)
        {
            tStep = tStepUpperBound;
        }

        return (tStep);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mStepLowerBound;
    ScalarType mStepUpperBound;
    ScalarType mContractionFactor;
    ScalarType mInitialTrialStepDotCurrentGradient;

    // mStepValues[0] = old trial value
    // mStepValues[1] = new trial value
    std::vector<ScalarType> mStepValues;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;

private:
    QuadraticLineSearch(const locus::QuadraticLineSearch<ScalarType, OrdinalType> & aRhs);
    locus::QuadraticLineSearch<ScalarType, OrdinalType> & operator=(const locus::QuadraticLineSearch<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_QUADRATICLINESEARCH_HPP_ */
