/*
 * Locus_SteihaugTointSolver.hpp
 *
 *  Created on: Oct 21, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef LOCUS_STEIHAUGTOINTSOLVER_HPP_
#define LOCUS_STEIHAUGTOINTSOLVER_HPP_

#include <cmath>
#include <limits>
#include <cassert>

#include "Locus_Vector.hpp"
#include "Locus_MultiVector.hpp"
#include "Locus_TrustRegionStageMng.hpp"
#include "Locus_TrustRegionAlgorithmDataMng.hpp"

namespace locus
{

struct krylov_solver
{
    enum stop_t
    {
        NaN_CURVATURE = 1,
        ZERO_CURVATURE = 2,
        NEGATIVE_CURVATURE = 3,
        INF_CURVATURE = 4,
        TOLERANCE = 5,
        TRUST_REGION_RADIUS = 6,
        MAX_ITERATIONS = 7,
        NaN_NORM_RESIDUAL = 8,
        INF_NORM_RESIDUAL = 9,
        INEXACTNESS_MEASURE = 10,
        ORTHOGONALITY_MEASURE = 11,
    };
};

template<typename ScalarType, typename OrdinalType = size_t>
class SteihaugTointSolver
{
public:
    SteihaugTointSolver() :
            mMaxNumIterations(200),
            mNumIterationsDone(0),
            mTolerance(1e-8),
            mNormResidual(0),
            mTrustRegionRadius(0),
            mRelativeTolerance(1e-1),
            mRelativeToleranceExponential(0.5),
            mStoppingCriterion(locus::krylov_solver::stop_t::MAX_ITERATIONS)
    {
    }
    virtual ~SteihaugTointSolver()
    {
    }

    void setMaxNumIterations(const OrdinalType & aInput)
    {
        mMaxNumIterations = aInput;
    }
    OrdinalType getMaxNumIterations() const
    {
        return (mMaxNumIterations);
    }
    void setNumIterationsDone(const OrdinalType & aInput)
    {
        mNumIterationsDone = aInput;
    }
    OrdinalType getNumIterationsDone() const
    {
        return (mNumIterationsDone);
    }
    void setSolverTolerance(const ScalarType & aInput)
    {
        mTolerance = aInput;
    }
    ScalarType getSolverTolerance() const
    {
        return (mTolerance);
    }
    void setTrustRegionRadius(const ScalarType & aInput)
    {
        mTrustRegionRadius = aInput;
    }
    ScalarType getTrustRegionRadius() const
    {
        return (mTrustRegionRadius);
    }
    void setNormResidual(const ScalarType & aInput)
    {
        mNormResidual = aInput;
    }
    ScalarType getNormResidual() const
    {
        return (mNormResidual);
    }
    void setRelativeTolerance(const ScalarType & aInput)
    {
        mRelativeTolerance = aInput;
    }
    ScalarType getRelativeTolerance() const
    {
        return (mRelativeTolerance);
    }
    void setRelativeToleranceExponential(const ScalarType & aInput)
    {
        mRelativeToleranceExponential = aInput;
    }
    ScalarType getRelativeToleranceExponential() const
    {
        return (mRelativeToleranceExponential);
    }
    void setStoppingCriterion(const locus::krylov_solver::stop_t & aInput)
    {
        mStoppingCriterion = aInput;
    }
    locus::krylov_solver::stop_t getStoppingCriterion() const
    {
        return (mStoppingCriterion);
    }
    ScalarType computeSteihaugTointStep(const locus::MultiVector<ScalarType, OrdinalType> & aNewtonStep,
                                        const locus::MultiVector<ScalarType, OrdinalType> & aConjugateDir,
                                        const locus::MultiVector<ScalarType, OrdinalType> & aPrecTimesNewtonStep,
                                        const locus::MultiVector<ScalarType, OrdinalType> & aPrecTimesConjugateDir)
    {
        assert(aNewtonStep.getNumVectors() == aConjugateDir.getNumVectors());
        assert(aNewtonStep.getNumVectors() == aPrecTimesNewtonStep.getNumVectors());
        assert(aNewtonStep.getNumVectors() == aPrecTimesConjugateDir.getNumVectors());

        // Dogleg trust region step
        OrdinalType tNumVectors = aNewtonStep.getNumVectors();
        ScalarType tNewtonStepDotPrecTimesNewtonStep = 0;
        ScalarType tNewtonStepDotPrecTimesConjugateDir = 0;
        ScalarType tConjugateDirDotPrecTimesConjugateDir = 0;
        for(OrdinalType tVectorIndex = 0; tVectorIndex < tNumVectors; tVectorIndex++)
        {
            tNewtonStepDotPrecTimesNewtonStep += aNewtonStep[tVectorIndex].dot(aPrecTimesNewtonStep[tVectorIndex]);
            tNewtonStepDotPrecTimesConjugateDir += aNewtonStep[tVectorIndex].dot(aPrecTimesConjugateDir[tVectorIndex]);
            tConjugateDirDotPrecTimesConjugateDir += aConjugateDir[tVectorIndex].dot(aPrecTimesConjugateDir[tVectorIndex]);
        }

        ScalarType tTrustRegionRadius = this->getTrustRegionRadius();
        ScalarType tAlpha = tNewtonStepDotPrecTimesConjugateDir * tNewtonStepDotPrecTimesConjugateDir;
        ScalarType tBeta = tConjugateDirDotPrecTimesConjugateDir
                * (tTrustRegionRadius * tTrustRegionRadius - tNewtonStepDotPrecTimesNewtonStep);
        ScalarType tAlphaPlusBeta = tAlpha + tBeta;
        ScalarType tNumerator = -tNewtonStepDotPrecTimesConjugateDir + std::sqrt(tAlphaPlusBeta);
        ScalarType tStep = tNumerator / tConjugateDirDotPrecTimesConjugateDir;

        return (tStep);
    }
    bool invalidCurvatureDetected(const ScalarType & aInput)
    {
        bool tInvalidCurvatureDetected = false;

        if(aInput < static_cast<ScalarType>(0.))
        {
            this->setStoppingCriterion(locus::krylov_solver::stop_t::NEGATIVE_CURVATURE);
            tInvalidCurvatureDetected = true;
        }
        else if(std::abs(aInput) <= std::numeric_limits<ScalarType>::min())
        {
            this->setStoppingCriterion(locus::krylov_solver::stop_t::ZERO_CURVATURE);
            tInvalidCurvatureDetected = true;
        }
        else if(std::isinf(aInput))
        {
            this->setStoppingCriterion(locus::krylov_solver::stop_t::INF_CURVATURE);
            tInvalidCurvatureDetected = true;
        }
        else if(std::isnan(aInput))
        {
            this->setStoppingCriterion(locus::krylov_solver::stop_t::NaN_CURVATURE);
            tInvalidCurvatureDetected = true;
        }

        return (tInvalidCurvatureDetected);
    }
    bool toleranceSatisfied(const ScalarType & aNormDescentDirection)
    {
        this->setNormResidual(aNormDescentDirection);
        ScalarType tStoppingTolerance = this->getSolverTolerance();

        bool tToleranceCriterionSatisfied = false;
        if(aNormDescentDirection < tStoppingTolerance)
        {
            this->setStoppingCriterion(locus::krylov_solver::stop_t::TOLERANCE);
            tToleranceCriterionSatisfied = true;
        }
        else if(std::isinf(aNormDescentDirection))
        {
            this->setStoppingCriterion(locus::krylov_solver::stop_t::INF_NORM_RESIDUAL);
            tToleranceCriterionSatisfied = true;
        }
        else if(std::isnan(aNormDescentDirection))
        {
            this->setStoppingCriterion(locus::krylov_solver::stop_t::NaN_NORM_RESIDUAL);
            tToleranceCriterionSatisfied = true;
        }

        return (tToleranceCriterionSatisfied);
    }

    virtual void solve(locus::TrustRegionStageMng<ScalarType, OrdinalType> & aStageMng,
                       locus::TrustRegionAlgorithmDataMng<ScalarType, OrdinalType> & aDataMng) = 0;

private:
    OrdinalType mMaxNumIterations;
    OrdinalType mNumIterationsDone;

    ScalarType mTolerance;
    ScalarType mNormResidual;
    ScalarType mTrustRegionRadius;
    ScalarType mRelativeTolerance;
    ScalarType mRelativeToleranceExponential;

    locus::krylov_solver::stop_t mStoppingCriterion;

private:
    SteihaugTointSolver(const locus::SteihaugTointSolver<ScalarType, OrdinalType> & aRhs);
    locus::SteihaugTointSolver<ScalarType, OrdinalType> & operator=(const locus::SteihaugTointSolver<ScalarType, OrdinalType> & aRhs);
};

}

#endif /* LOCUS_STEIHAUGTOINTSOLVER_HPP_ */
