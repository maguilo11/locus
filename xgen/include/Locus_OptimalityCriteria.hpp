/*
 * Locus_OptimalityCriteria.hpp
 *
 *  Created on: Oct 17, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_OPTIMALITYCRITERIA_HPP_
#define LOCUS_OPTIMALITYCRITERIA_HPP_

#include <sstream>
#include <iostream>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_OptimalityCriteriaDataMng.hpp"
#include "Locus_OptimalityCriteriaSubProblem.hpp"
#include "Locus_OptimalityCriteriaStageMngBase.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class OptimalityCriteria
{
public:
    explicit OptimalityCriteria(const std::shared_ptr<locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType>> & aDataMng,
                                const std::shared_ptr<locus::OptimalityCriteriaStageMngBase<ScalarType, OrdinalType>> & aStageMng,
                                const std::shared_ptr<locus::OptimalityCriteriaSubProblem<ScalarType, OrdinalType>> & aSubProblem) :
            mPrintDiagnostics(false),
            mOutputStream(),
            mMaxNumIterations(50),
            mNumIterationsDone(0),
            mStagnationTolerance(1e-2),
            mFeasibilityTolerance(1e-5),
            mObjectiveGradientTolerance(1e-8),
            mDataMng(aDataMng),
            mStageMng(aStageMng),
            mSubProblem(aSubProblem)
    {
    }
    ~OptimalityCriteria()
    {
    }

    bool printDiagnostics() const
    {
        return (mPrintDiagnostics);
    }

    void enableDiagnostics()
    {
        mPrintDiagnostics = true;
    }

    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    ScalarType getStagnationTolerance() const
    {
        return (mStagnationTolerance);
    }
    ScalarType getFeasibilityTolerance() const
    {
        return (mFeasibilityTolerance);
    }
    ScalarType getObjectiveGradientTolerance() const
    {
        return (mObjectiveGradientTolerance);
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    void setStagnationTolerance(const ScalarType & aInput)
    {
        mStagnationTolerance = aInput;
    }
    void setFeasibilityTolerance(const ScalarType & aInput)
    {
        mFeasibilityTolerance = aInput;
    }
    void setObjectiveGradientTolerance(const ScalarType & aInput)
    {
        mObjectiveGradientTolerance = aInput;
    }

    void gatherOuputStream(std::ostringstream & aOutput)
    {
        aOutput << mOutputStream.str().c_str();
    }

    void solve()
    {
        const locus::MultiVector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl();
        for(OrdinalType tConstraintIndex = 0; tConstraintIndex < mDataMng->getNumConstraints(); tConstraintIndex++)
        {
            ScalarType tValue = mStageMng->evaluateInequality(tConstraintIndex, tControl);
            mDataMng->setCurrentConstraintValue(tConstraintIndex, tValue);
        }

        while(1)
        {
            mStageMng->update(*mDataMng);

            mDataMng->computeStagnationMeasure();
            mDataMng->computeMaxInequalityValue();
            mDataMng->computeNormObjectiveGradient();

            this->printCurrentProgress();
            if(this->stoppingCriteriaSatisfied() == true)
            {
                break;
            }

            this->storeCurrentStageData();
            mSubProblem->solve(*mDataMng, *mStageMng);

            mNumIterationsDone++;
        }
    }

    bool stoppingCriteriaSatisfied()
    {
        bool tStoppingCriterionSatisfied = false;
        ScalarType tStagnationMeasure = mDataMng->getStagnationMeasure();
        ScalarType tMaxInequalityValue = mDataMng->getMaxInequalityValue();
        ScalarType tNormObjectiveGradient = mDataMng->getNormObjectiveGradient();

        if(this->getNumIterationsDone() >= this->getMaxNumIterations())
        {
            tStoppingCriterionSatisfied = true;
        }
        else if(tStagnationMeasure < this->getStagnationTolerance())
        {
            tStoppingCriterionSatisfied = true;
        }
        else if(tNormObjectiveGradient < this->getObjectiveGradientTolerance()
                && tMaxInequalityValue < this->getFeasibilityTolerance())
        {
            tStoppingCriterionSatisfied = true;
        }

        return (tStoppingCriterionSatisfied);
    }

    void printCurrentProgress()
    {
        if(this->printDiagnostics() == false)
        {
            return;
        }

        OrdinalType tCurrentNumIterationsDone = this->getNumIterationsDone();

        if(tCurrentNumIterationsDone < 2)
        {
            mOutputStream << " Itr" << std::setw(14) << "   F(x)  " << std::setw(16) << " ||F'(x)||" << std::setw(16)
                    << "   Max(H) " << "\n" << std::flush;
            mOutputStream << "-----" << std::setw(14) << "----------" << std::setw(16) << "-----------" << std::setw(16)
                    << "----------" << "\n" << std::flush;
        }

        ScalarType tObjectiveValue = mDataMng->getCurrentObjectiveValue();
        ScalarType tMaxInequalityValue = mDataMng->getMaxInequalityValue();
        ScalarType tNormObjectiveGradient = mDataMng->getNormObjectiveGradient();
        mOutputStream << std::setw(3) << tCurrentNumIterationsDone << std::setprecision(4) << std::fixed
                << std::scientific << std::setw(16) << tObjectiveValue << std::setw(16) << tNormObjectiveGradient
                << std::setw(16) << tMaxInequalityValue << "\n";
    }
    void storeCurrentStageData()
    {
        const ScalarType tObjectiveValue = mDataMng->getCurrentObjectiveValue();
        mDataMng->setPreviousObjectiveValue(tObjectiveValue);

        OrdinalType tNumVectors = mDataMng->getNumControlVectors();
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            const locus::Vector<ScalarType, OrdinalType> & tControl = mDataMng->getCurrentControl(tVectorIndex);
            mDataMng->setPreviousControl(tVectorIndex, tControl);
        }
    }

private:
    bool mPrintDiagnostics;
    std::ostringstream mOutputStream;

    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mStagnationTolerance;
    ScalarType mFeasibilityTolerance;
    ScalarType mObjectiveGradientTolerance;

    std::shared_ptr<locus::OptimalityCriteriaDataMng<ScalarType, OrdinalType>> mDataMng;
    std::shared_ptr<locus::OptimalityCriteriaStageMngBase<ScalarType, OrdinalType>> mStageMng;
    std::shared_ptr<locus::OptimalityCriteriaSubProblem<ScalarType, OrdinalType>> mSubProblem;

private:
    OptimalityCriteria(const locus::OptimalityCriteria<ScalarType, OrdinalType>&);
    locus::OptimalityCriteria<ScalarType, OrdinalType> & operator=(const locus::OptimalityCriteria<ScalarType, OrdinalType>&);
};

}

#endif /* LOCUS_OPTIMALITYCRITERIA_HPP_ */
