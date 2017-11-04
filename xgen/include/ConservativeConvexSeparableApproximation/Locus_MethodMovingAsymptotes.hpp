/*
 * Locus_MethodMovingAsymptotes.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_METHODMOVINGASYMPTOTES_HPP_
#define LOCUS_METHODMOVINGASYMPTOTES_HPP_

#include <memory>

#include "Locus_Bounds.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_PrimalProblemStageMng.hpp"
#include "Locus_NonlinearConjugateGradientDualSolver.hpp"
#include "Locus_ConservativeConvexSeparableAppxDataMng.hpp"
#include "Locus_ConservativeConvexSeparableApproximation.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class MethodMovingAsymptotes : public locus::ConservativeConvexSeparableApproximation<ScalarType, OrdinalType>
{
public:
    explicit MethodMovingAsymptotes(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory) :
            mTrialDual(aDataFactory.dual().create()),
            mActiveSet(aDataFactory.control().create()),
            mInactiveSet(aDataFactory.control().create()),
            mTrialControl(aDataFactory.control().create()),
            mConstraintValues(aDataFactory.dual().create()),
            mDualSolver(std::make_shared<locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType>>(aDataFactory))
    {
    }
    virtual ~MethodMovingAsymptotes()
    {
    }

    void solve(locus::PrimalProblemStageMng<ScalarType, OrdinalType> & aPrimalProblemStageMng,
               locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        // NOTE: REMBER THAT THE GLOBALIZATION FACTORS FOR BOTH OBJECTIVE AND CONSTRAINTS ARE SET TO ZERO IF USING
        //       THE MMA APPROACH. THE MMA METHOD IS DETECTED AT THE OUTTER LOOP LEVEL (NOT THE SUBPROBLEM LEVEL)
        //       AND THE VALUES SET INSIDE THE INITIALZE FUNCTION IN THE ALGORITHM CLASS. DEFAULT VALUES ARE ONLY
        //       SET FOR THE GCMMA CASE.
        mDualSolver->update(aDataMng);
        mDualSolver->updateObjectiveCoefficients(aDataMng);
        mDualSolver->updateConstraintCoefficients(aDataMng);
        mDualSolver->solve(mTrialDual.operator*(), mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = aDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = aDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());
        locus::bounds::computeActiveAndInactiveSets(mTrialControl.operator*(),
                                                    tLowerBounds,
                                                    tUpperBounds,
                                                    mActiveSet.operator*(),
                                                    mInactiveSet.operator*());
        aDataMng.setActiveSet(mActiveSet.operator*());
        aDataMng.setInactiveSet(mInactiveSet.operator*());

        ScalarType tObjectiveFunctionValue = aPrimalProblemStageMng.evaluateObjective(mTrialControl.operator*());
        aDataMng.setCurrentObjectiveFunctionValue(tObjectiveFunctionValue);
        aPrimalProblemStageMng.evaluateConstraints(mTrialControl.operator*(), mConstraintValues.operator*());
        aDataMng.setCurrentConstraintValues(mConstraintValues.operator*());
        aDataMng.setCurrentControl(mTrialControl.operator*());
        aDataMng.setDual(mTrialDual.operator*());
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualSolver->initializeAuxiliaryVariables(aDataMng);
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialDual;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mActiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mInactiveSet;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mConstraintValues;
    std::shared_ptr<locus::DualProblemSolver<ScalarType, OrdinalType>> mDualSolver;

private:
    MethodMovingAsymptotes(const locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
    locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & operator=(const locus::MethodMovingAsymptotes<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_METHODMOVINGASYMPTOTES_HPP_ */
