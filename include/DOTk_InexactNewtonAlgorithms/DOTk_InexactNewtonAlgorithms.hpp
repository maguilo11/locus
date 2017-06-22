/*
 * DOTk_InexactNewtonAlgorithms.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEXACTNEWTONALGORITHMS_HPP_
#define DOTK_INEXACTNEWTONALGORITHMS_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_KrylovSolver;
class DOTk_OptimizationDataMng;
class DOTk_KrylovSolverStoppingCriterion;

class DOTk_InexactNewtonAlgorithms
{
public:
    explicit DOTk_InexactNewtonAlgorithms(dotk::types::algorithm_t type_);
    virtual ~DOTk_InexactNewtonAlgorithms();

    void setMaxNumItr(size_t itr_);
    size_t getMaxNumItr() const;
    virtual void setNumItrDone(size_t itr_);
    size_t getNumItrDone() const;

    void setRelativeTolerance(Real tolerance_);
    void setObjectiveFuncTol(Real tol_);
    Real getObjectiveFuncTol() const;
    void setGradientTol(Real tol_);
    Real getGradientTol() const;
    void setTrialStepTol(Real tol_);
    Real getTrialStepTol() const;
    void setMinCosineAngleTol(Real tol_);
    Real getMinCosineAngleTol() const;

    dotk::types::algorithm_t type() const;
    void setStoppingCriterion(dotk::types::stop_criterion_t flag_);
    dotk::types::stop_criterion_t getStoppingCriterion() const;

    void setFixedStoppingCriterion(Real fixed_tolerance_);
    void setRelativeStoppingCriterion(Real relative_tolerance_);
    void setTrialStep(const std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                      const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);
    bool checkStoppingCriteria(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

    virtual void setMaxNumKrylovSolverItr(size_t itr_) = 0;

protected:
    std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> m_Criterion;

private:
    void resetCurrentStateToFormer(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    size_t m_MaxNumOptItr;
    size_t m_NumOptItrDone;

    Real m_FvalTol;
    Real m_GradTol;
    Real m_StepTol;
    Real m_MinCosineAngleTol;

    dotk::types::algorithm_t m_AlgorithmType;
    dotk::types::stop_criterion_t m_StoppingCriterion;

private:
    DOTk_InexactNewtonAlgorithms(const dotk::DOTk_InexactNewtonAlgorithms&);
    dotk::DOTk_InexactNewtonAlgorithms & operator=(const dotk::DOTk_InexactNewtonAlgorithms &);
};

}

#endif /* DOTK_INEXACTNEWTONALGORITHMS_HPP_ */
