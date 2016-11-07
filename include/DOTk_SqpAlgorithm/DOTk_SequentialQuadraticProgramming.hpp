/*
 * DOTk_SequentialQuadraticProgramming.hpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_SEQUENTIALQUADRATICPROGRAMMING_HPP_
#define DOTK_SEQUENTIALQUADRATICPROGRAMMING_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_TrustRegionMngTypeELP;

class DOTk_SequentialQuadraticProgramming
{
public:
    explicit DOTk_SequentialQuadraticProgramming(dotk::types::algorithm_t type_);
    virtual ~DOTk_SequentialQuadraticProgramming();

    void setMaxNumItr(size_t itr_);
    size_t getMaxNumItr() const;
    void setNumItrDone(size_t itr_);
    size_t getNumItrDone() const;
    void setNumTrustRegionSubProblemItrDone(size_t itr_);
    size_t getNumTrustRegionSubProblemItrDone() const;

    void setGradientTolerance(Real tol_);
    Real getGradientTolerance() const;
    void setTrialStepTolerance(Real tol_);
    Real getTrialStepTolerance() const;
    void setOptimalityTolerance(Real tol_);
    Real getOptimalityTolerance() const;
    void setFeasibilityTolerance(Real tol_);
    Real getFeasibilityTolerance() const;

    void setStoppingCriterion(dotk::types::stop_criterion_t flag_);
    dotk::types::stop_criterion_t getStoppingCriterion() const;

    bool checkStoppingCriteria(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    void storePreviousSolution(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);

private:
    void resetCurrentStateToFormer(const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);

private:
    size_t m_MaxNumOptItr;
    size_t m_NumOptItrDone;
    size_t m_NumTrustRegionSubProblemItrDone;

    Real m_GradientTolerance;
    Real m_TrialStepTolerance;
    Real m_OptimalityTolerance;
    Real m_FeasibilityTolerance;

    dotk::types::algorithm_t m_AlgorithmType;
    dotk::types::stop_criterion_t m_StoppingCriterion;

private:
    DOTk_SequentialQuadraticProgramming(const dotk::DOTk_SequentialQuadraticProgramming&);
    dotk::DOTk_SequentialQuadraticProgramming operator=(const dotk::DOTk_SequentialQuadraticProgramming&);
};

}

#endif /* DOTK_SEQUENTIALQUADRATICPROGRAMMING_HPP_ */
