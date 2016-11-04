/*
 * DOTk_FirstOrderAlgorithm.hpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_FIRSTORDERALGORITHM_HPP_
#define DOTK_FIRSTORDERALGORITHM_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

class DOTk_FirstOrderAlgorithm
{
public:
    DOTk_FirstOrderAlgorithm();
    explicit DOTk_FirstOrderAlgorithm(dotk::types::algorithm_t type_);
    ~DOTk_FirstOrderAlgorithm();

    void setMaxNumItr(size_t itr_);
    size_t getMaxNumItr() const;
    void setNumItrDone(size_t itr_);
    size_t getNumItrDone() const;
    void setObjectiveFuncTol(Real tol_);
    Real getObjectiveFuncTol() const;
    void setGradientTol(Real tol_);
    Real getGradientTol() const;
    void setTrialStepTol(Real tol_);
    Real getTrialStepTol() const;
    void setMinCosineAngleTol(Real tol_);
    Real getMinCosineAngleTol() const;

    void setAlgorithmType(dotk::types::algorithm_t type_);
    dotk::types::algorithm_t getAlgorithmType() const;
    void setStoppingCriterion(dotk::types::stop_criterion_t flag_);
    dotk::types::stop_criterion_t getStoppingCriterion() const;

    bool checkStoppingCriteria(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    void resetCurrentStateToFormer(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    size_t mMaxNumItr;
    size_t mNumItrDone;

    Real mFvalTol;
    Real mGradTol;
    Real mStepTol;
    Real mMinCosineAngleTol;

    dotk::types::algorithm_t mAlgorithmType;
    dotk::types::stop_criterion_t m_StoppingCriterion;

private:
    DOTk_FirstOrderAlgorithm(const dotk::DOTk_FirstOrderAlgorithm &);
    DOTk_FirstOrderAlgorithm & operator=(const dotk::DOTk_FirstOrderAlgorithm &);
};

}

#endif /* DOTK_FIRSTORDERALGORITHM_HPP_ */
