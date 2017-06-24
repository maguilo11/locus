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
    explicit DOTk_InexactNewtonAlgorithms(dotk::types::algorithm_t aType);
    virtual ~DOTk_InexactNewtonAlgorithms();

    void setMaxNumItr(size_t aInput);
    size_t getMaxNumItr() const;
    virtual void setNumItrDone(size_t aInput);
    size_t getNumItrDone() const;

    void setRelativeTolerance(Real aInput);
    void setObjectiveFuncTol(Real aInput);
    Real getObjectiveFuncTol() const;
    void setGradientTol(Real aInput);
    Real getGradientTol() const;
    void setTrialStepTol(Real aInput);
    Real getTrialStepTol() const;
    void setMinCosineAngleTol(Real aInput);
    Real getMinCosineAngleTol() const;

    dotk::types::algorithm_t type() const;
    void setStoppingCriterion(dotk::types::stop_criterion_t aInput);
    dotk::types::stop_criterion_t getStoppingCriterion() const;

    void setFixedStoppingCriterion(Real aInput);
    void setRelativeStoppingCriterion(Real aInput);
    void setTrialStep(const std::shared_ptr<dotk::DOTk_KrylovSolver> & aSolver,
                      const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);
    bool checkStoppingCriteria(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

    virtual void setMaxNumKrylovSolverItr(size_t aInput) = 0;

protected:
    std::shared_ptr<dotk::DOTk_KrylovSolverStoppingCriterion> m_Criterion;

private:
    void resetCurrentStateToFormer(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng);

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
