/*
 * Locus_NonlinearConjugateGradient.hpp
 *
 *  Created on: Oct 22, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_NONLINEARCONJUGATEGRADIENT_HPP_
#define LOCUS_NONLINEARCONJUGATEGRADIENT_HPP_

#include <limits>
#include <memory>
#include <cassert>

#include "Locus_Types.hpp"
#include "Locus_Bounds.hpp"
#include "Locus_Daniels.hpp"
#include "Locus_DaiLiao.hpp"
#include "Locus_DaiYuan.hpp"
#include "Locus_LiuStorey.hpp"
#include "Locus_HagerZhang.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_DataFactory.hpp"
#include "Locus_PerryShanno.hpp"
#include "Locus_PolakRibiere.hpp"
#include "Locus_DaiYuanHybrid.hpp"
#include "Locus_LinearAlgebra.hpp"
#include "Locus_FletcherReeves.hpp"
#include "Locus_HestenesStiefel.hpp"
#include "Locus_CubicLineSearch.hpp"
#include "Locus_ConjugateDescent.hpp"
#include "Locus_QuadraticLineSearch.hpp"
#include "Locus_NonlinearConjugateGradientDataMng.hpp"
#include "Locus_NonlinearConjugateGradientStateMng.hpp"
#include "Locus_NonlinearConjugateGradientStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class NonlinearConjugateGradient
{
public:
    NonlinearConjugateGradient(const std::shared_ptr<locus::DataFactory<ScalarType, OrdinalType>> & aDataFactory,
                               const std::shared_ptr<locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType>> & aDataMng,
                               const std::shared_ptr<locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType>> & aStageMng) :
            mMaxNumIterations(500),
            mNumIterationsDone(0),
            mGradientTolerance(1e-8),
            mStationarityTolerance(1e-8),
            mControlStagnationTolerance(std::numeric_limits<ScalarType>::epsilon()),
            mObjectiveStagnationTolerance(std::numeric_limits<ScalarType>::epsilon()),
            mStoppingCriteria(locus::algorithm::stop_t::NOT_CONVERGED),
            mControlWork(aDataFactory->control().create()),
            mTrialControl(aDataFactory->control().create()),
            mLineSearch(std::make_shared<locus::CubicLineSearch<ScalarType, OrdinalType>>(aDataFactory.operator*())),
            mStep(std::make_shared<locus::PolakRibiere<ScalarType, OrdinalType>>(aDataFactory.operator*())),
            mStateMng(std::make_shared<locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType>>(aDataMng, aStageMng))
    {
    }
    ~NonlinearConjugateGradient()
    {
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    locus::algorithm::stop_t getStoppingCriteria() const
    {
        return (mStoppingCriteria);
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setGradientTolerance(const ScalarType & aInput)
    {
        mGradientTolerance = aInput;
    }
    void setStationarityTolerance(const ScalarType & aInput)
    {
        mStationarityTolerance = aInput;
    }
    void setControlStagnationTolerance(const ScalarType & aInput)
    {
        mControlStagnationTolerance = aInput;
    }
    void setObjectiveStagnationTolerance(const ScalarType & aInput)
    {
        mObjectiveStagnationTolerance = aInput;
    }

    void setContractionFactor(const ScalarType & aInput)
    {
        mLineSearch->setContractionFactor(aInput);
    }
    void setDanielsMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::Daniels<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiLiaoMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiLiao<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiYuanMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiYuan<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setLiuStoreyMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::LiuStorey<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setHagerZhangMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::HagerZhang<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setPerryShannoMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::PerryShanno<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setDaiYuanHybridMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::DaiYuanHybrid<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setFletcherReevesMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::FletcherReeves<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setHestenesStiefelMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::HestenesStiefel<ScalarType, OrdinalType>>(aDataFactory);
    }
    void setConjugateDescentMethod(const locus::DataFactory<ScalarType, OrdinalType> & aDataFactory)
    {
        mStep = std::make_shared<locus::ConjugateDescent<ScalarType, OrdinalType>>(aDataFactory);
    }

    void solve()
    {
        assert(mStep.get() != nullptr);

        this->computeInitialState();
        // Perform first iteration (i.e. x_0)
        this->computeInitialDescentDirection();
        this->computeProjectedStep();
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        tDataMng.storePreviousState();

        mLineSearch->step(mStateMng.operator*());
        locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
        mStateMng->computeGradient(tDataMng.getCurrentControl(), mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
        this->computeProjectedGradient();

        bool tStop = false;
        if(this->checkStoppingCriteria() == true)
        {
            tStop = true;
        }

        mNumIterationsDone = 1;
        locus::NonlinearConjugateGradientStageMngBase<ScalarType, OrdinalType> & tStageMng = mStateMng->getStageMng();
        while(tStop != true)
        {
            mStep->computeScaledDescentDirection(tDataMng, tStageMng);
            this->computeProjectedStep();
            tDataMng.storePreviousState();

            mLineSearch->step(mStateMng.operator*());
            locus::fill(static_cast<ScalarType>(0), mControlWork.operator*());
            mStateMng->computeGradient(tDataMng.getCurrentControl(), mControlWork.operator*());
            tDataMng.setCurrentGradient(mControlWork.operator*());
            this->computeProjectedGradient();

            mNumIterationsDone++;
            if(this->checkStoppingCriteria() == true)
            {
                tStop = true;
                break;
            }
        }
    }

private:
    void computeInitialState()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        ScalarType tValue = mStateMng->evaluateObjective(tControl);
        tDataMng.setCurrentObjectiveFunctionValue(tValue);

        mStateMng->computeGradient(tControl, mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
        this->computeProjectedGradient();
    }
    void computeProjectedStep()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tControl, static_cast<ScalarType>(0), mTrialControl.operator*());
        locus::update(static_cast<ScalarType>(1),
                      tDataMng.getTrialStep(),
                      static_cast<ScalarType>(1),
                      mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = tDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = tDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());

        // Compute projected trial step
        locus::update(static_cast<ScalarType>(1),
                      mTrialControl.operator*(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        locus::update(static_cast<ScalarType>(-1), tControl, static_cast<ScalarType>(1), mControlWork.operator*());
        tDataMng.setTrialStep(mControlWork.operator*());
    }
    void computeProjectedGradient()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = tDataMng.getCurrentControl();
        locus::update(static_cast<ScalarType>(1), tControl, static_cast<ScalarType>(0), mTrialControl.operator*());
        locus::update(static_cast<ScalarType>(-1),
                      tDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(1),
                      mTrialControl.operator*());

        const locus::MultiVector<ScalarType, OrdinalType> & tLowerBounds = tDataMng.getControlLowerBounds();
        const locus::MultiVector<ScalarType, OrdinalType> & tUpperBounds = tDataMng.getControlUpperBounds();
        locus::bounds::project(tLowerBounds, tUpperBounds, mTrialControl.operator*());

        // Compute projected gradient
        locus::update(static_cast<ScalarType>(-1),
                      mTrialControl.operator*(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        locus::update(static_cast<ScalarType>(1), tControl, static_cast<ScalarType>(1), mControlWork.operator*());
        tDataMng.setCurrentGradient(mControlWork.operator*());
    }
    void computeInitialDescentDirection()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();
        locus::update(static_cast<ScalarType>(-1),
                      tDataMng.getCurrentGradient(),
                      static_cast<ScalarType>(0),
                      mControlWork.operator*());
        tDataMng.setTrialStep(mControlWork.operator*());
    }
    bool checkStoppingCriteria()
    {
        locus::NonlinearConjugateGradientDataMng<ScalarType, OrdinalType> & tDataMng = mStateMng->getDataMng();

        tDataMng.computeNormGradient();
        tDataMng.computeStagnationMeasure();
        tDataMng.computeStationarityMeasure();
        tDataMng.computeObjectiveStagnationMeasure();

        const ScalarType tNormGradient = tDataMng.getNormGradient();
        const ScalarType tStagnationMeasure = tDataMng.getStagnationMeasure();
        const ScalarType tStationarityMeasure = tDataMng.getStationarityMeasure();
        const ScalarType tObjectiveStagnationMeasure = tDataMng.getObjectiveStagnationMeasure();

        bool tStop = false;
        if(tStagnationMeasure < mControlStagnationTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::CONTROL_STAGNATION;
        }
        else if(tNormGradient < mGradientTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::NORM_GRADIENT;
        }
        else if(tStationarityMeasure < mStationarityTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::NORM_STEP;
        }
        else if(mNumIterationsDone >= mMaxNumIterations)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::MAX_NUMBER_ITERATIONS;
        }
        else if(tObjectiveStagnationMeasure < mObjectiveStagnationTolerance)
        {
            tStop = true;
            mStoppingCriteria = locus::algorithm::stop_t::OBJECTIVE_STAGNATION;
        }

        return (tStop);
    }

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mGradientTolerance;
    ScalarType mStationarityTolerance;
    ScalarType mControlStagnationTolerance;
    ScalarType mObjectiveStagnationTolerance;

    locus::algorithm::stop_t mStoppingCriteria;

    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mControlWork;
    std::shared_ptr<locus::MultiVector<ScalarType, OrdinalType>> mTrialControl;

    std::shared_ptr<locus::LineSearch<ScalarType, OrdinalType>> mLineSearch;
    std::shared_ptr<locus::NonlinearConjugateGradientStep<ScalarType, OrdinalType>> mStep;
    std::shared_ptr<locus::NonlinearConjugateGradientStateMng<ScalarType, OrdinalType>> mStateMng;

private:
    NonlinearConjugateGradient(const locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & aRhs);
    locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & operator=(const locus::NonlinearConjugateGradient<ScalarType, OrdinalType> & aRhs);
};

} // namespace locus

#endif /* LOCUS_NONLINEARCONJUGATEGRADIENT_HPP_ */
