/*
 * Locus_NonlinearConjugateGradientDualSolver.hpp
 *
 *  Created on: Nov 4, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_NONLINEARCONJUGATEGRADIENTDUALSOLVER_HPP_
#define LOCUS_NONLINEARCONJUGATEGRADIENTDUALSOLVER_HPP_

#include <limits>
#include <memory>

#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_DualProblemSolver.hpp"
#include "Locus_DualProblemStageMng.hpp"
#include "Locus_NonlinearConjugateGradient.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_ConservativeConvexSeparableAppxDataMng.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradientDualSolver : public locus::DualProblemSolver<ScalarType, OrdinalType>
{
public:
    explicit NonlinearConjugateGradientDualSolver(const locus::DataFactory<ScalarType, OrdinalType> & aPrimalDataFactory) :
            mDualWork(aPrimalDataFactory.dual().create()),
            mDualInitialGuess(aPrimalDataFactory.dual().create()),
            mDualDataFactory(std::make_shared<locus::DataFactory<ScalarType, OrdinalType>>()),
            mDualStageMng(),
            mDualAlgorithm(),
            mDualDataMng()
    {
        this->initialize(aPrimalDataFactory);
    }
    virtual ~NonlinearConjugateGradientDualSolver()
    {
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mDualAlgorithm->setMaxNumIterations(aInput);
    }
    void setContractionFactor(const ScalarType & aInput)
    {
        mDualAlgorithm->setContractionFactor(aInput);
    }
    void setGradientTolerance(const ScalarType & aInput)
    {
        mDualAlgorithm->setGradientTolerance(aInput);
    }
    void setStationarityTolerance(const ScalarType & aInput)
    {
        mDualAlgorithm->setStationarityTolerance(aInput);
    }
    void setControlStagnationTolerance(const ScalarType & aInput)
    {
        mDualAlgorithm->setControlStagnationTolerance(aInput);
    }
    void setObjectiveStagnationTolerance(const ScalarType & aInput)
    {
        mDualAlgorithm->setObjectiveStagnationTolerance(aInput);
    }

    void solve(locus::MultiVector<ScalarType, OrdinalType> & aDual,
               locus::MultiVector<ScalarType, OrdinalType> & aTrialControl)
    {
        this->reset();
        //mDualDataMng->setInitialGuess(mDualInitialGuess.operator*());
        mDualAlgorithm->solve();

        mDualStageMng->getTrialControl(aTrialControl);
        const locus::MultiVector<ScalarType, OrdinalType> & tDualSolution = mDualDataMng->getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tDualSolution, static_cast<ScalarType>(0), aDual);
        // Store dual solution and use it as the initial guess for the next iteration.
        locus::update(static_cast<ScalarType>(1), tDualSolution, static_cast<ScalarType>(0), mDualInitialGuess.operator*());
    }

    void update(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->update(aDataMng);
    }
    void updateObjectiveCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->updateObjectiveCoefficients(aDataMng);
    }
    void updateConstraintCoefficients(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->updateConstraintCoefficients(aDataMng);
    }
    void initializeAuxiliaryVariables(locus::ConservativeConvexSeparableAppxDataMng<ScalarType, OrdinalType> & aDataMng)
    {
        mDualStageMng->initializeAuxiliaryVariables(aDataMng);
    }

private:
    void reset()
    {
        const ScalarType tValue = 0;
        mDualDataMng->setCurrentObjectiveFunctionValue(tValue);
        mDualDataMng->setPreviousObjectiveFunctionValue(tValue);

        locus::fill(tValue, mDualWork.operator*());
        mDualDataMng->setTrialStep(mDualWork.operator*());
        mDualDataMng->setCurrentControl(mDualWork.operator*());
        mDualDataMng->setPreviousControl(mDualWork.operator*());
        mDualDataMng->setCurrentGradient(mDualWork.operator*());
        mDualDataMng->setPreviousGradient(mDualWork.operator*());
    }
    void initialize(const locus::DataFactory<ScalarType, OrdinalType> & aPrimalDataFactory)
    {
        mDualDataFactory->allocateControl(mDualWork.operator*());
        mDualDataMng = std::make_shared<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>>(mDualDataFactory.operator*());

        ScalarType tValue = 0;
        mDualDataMng->setControlLowerBounds(tValue);
        tValue = std::numeric_limits<ScalarType>::max();
        mDualDataMng->setControlUpperBounds(tValue);

        mDualStageMng = std::make_shared<locus::DualProblemStageMng<ScalarType, OrdinalType>>(aPrimalDataFactory);
        mDualAlgorithm = std::make_shared<locus::NonlinearConjugateGradient<ScalarType, OrdinalType>>(mDualDataFactory, mDualDataMng, mDualStageMng);
        mDualAlgorithm->setFletcherReevesMethod(mDualDataFactory.operator*());

        OrdinalType tMaxIterations = 100;
        mDualAlgorithm->setMaxNumIterations(tMaxIterations);

        ScalarType tTolerance = 1e-7;
        mDualAlgorithm->setGradientTolerance(tTolerance);
        mDualAlgorithm->setStationarityTolerance(tTolerance);
    }

private:
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mDualInitialGuess;

    std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> mDualDataFactory;
    std::shared_ptr<locus::DualProblemStageMng<ScalarType, OrdinalType>> mDualStageMng;
    std::shared_ptr<locus::NonlinearConjugateGradient<ScalarType, OrdinalType>> mDualAlgorithm;
    std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> mDualDataMng;

private:
    NonlinearConjugateGradientDualSolver(const locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradientDualSolver<ScalarType, OrdinalType> & aRhs);
};

}

#endif /* LOCUS_NONLINEARCONJUGATEGRADIENTDUALSOLVER_HPP_ */
