/*
 * DOTk_OptimizationDataMng.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "vector.hpp"
#include "DOTk_Primal.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_OptimizationDataMng::DOTk_OptimizationDataMng() :
        m_NumIterations(0),
        m_NormTrialStep(0),
        m_NormNewGradient(0),
        m_OldObjectiveFunction(0),
        m_NewObjectiveFunction(0),
        m_OldDual(),
        m_NewDual(),
        m_TrialStep(),
        m_OldPrimal(),
        m_NewPrimal(),
        m_OldGradient(),
        m_NewGradient(),
        m_MatrixTimesVector()
{
}

DOTk_OptimizationDataMng::DOTk_OptimizationDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_) :
        m_NumIterations(0),
        m_NormTrialStep(0),
        m_NormNewGradient(0),
        m_OldObjectiveFunction(0),
        m_NewObjectiveFunction(0),
        m_OldDual(),
        m_NewDual(),
        m_TrialStep(primal_->control()->clone()),
        m_OldPrimal(primal_->control()->clone()),
        m_NewPrimal(primal_->control()->clone()),
        m_OldGradient(primal_->control()->clone()),
        m_NewGradient(primal_->control()->clone()),
        m_MatrixTimesVector(primal_->control()->clone())
{
    this->initialize(primal_);
}

DOTk_OptimizationDataMng::~DOTk_OptimizationDataMng()
{
}

void DOTk_OptimizationDataMng::setNumOptimizationItrDone(size_t input_)
{
    m_NumIterations= input_;
}

size_t DOTk_OptimizationDataMng::getNumOptimizationItrDone() const
{
    return (m_NumIterations);
}

void DOTk_OptimizationDataMng::setNormTrialStep(Real input_)
{
    m_NormTrialStep = input_;
}

Real DOTk_OptimizationDataMng::getNormTrialStep() const
{
    return (m_NormTrialStep);
}

void DOTk_OptimizationDataMng::setNormNewGradient(Real input_)
{
    m_NormNewGradient = input_;
}

Real DOTk_OptimizationDataMng::getNormNewGradient() const
{
    return (m_NormNewGradient);
}

void DOTk_OptimizationDataMng::storeCurrentState()
{
    m_OldObjectiveFunction = m_NewObjectiveFunction;
    m_OldPrimal->copy(*m_NewPrimal);
    m_OldGradient->copy(*m_NewGradient);
}

void DOTk_OptimizationDataMng::setNewObjectiveFunctionValue(Real value_)
{
    m_NewObjectiveFunction = value_;
}

Real DOTk_OptimizationDataMng::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFunction);
}

void DOTk_OptimizationDataMng::setOldObjectiveFunctionValue(Real value_)
{
    m_OldObjectiveFunction = value_;
}

Real DOTk_OptimizationDataMng::getOldObjectiveFunctionValue() const
{
    return (m_OldObjectiveFunction);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getOldDual() const
{
    return (m_OldDual);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getNewDual() const
{
    return (m_NewDual);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getMatrixTimesVector() const
{
    return (m_MatrixTimesVector);
}

void DOTk_OptimizationDataMng::setTrialStep(const dotk::vector<Real> & input_)
{
    m_TrialStep->copy(input_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getTrialStep() const
{
    return (m_TrialStep);
}

void DOTk_OptimizationDataMng::setNewPrimal(const dotk::vector<Real> & input_)
{
    m_NewPrimal->copy(input_);
}

void DOTk_OptimizationDataMng::setOldPrimal(const dotk::vector<Real> & input_)
{
    m_OldPrimal->copy(input_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getNewPrimal() const
{
    return (m_NewPrimal);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getOldPrimal() const
{
    return (m_OldPrimal);
}

void DOTk_OptimizationDataMng::setNewGradient(const dotk::vector<Real> & input_)
{
    m_NewGradient->copy(input_);
}

void DOTk_OptimizationDataMng::setOldGradient(const dotk::vector<Real> & input_)
{
    m_OldGradient->copy(input_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getNewGradient() const
{
    return (m_NewGradient);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_OptimizationDataMng::getOldGradient() const
{
    return (m_OldGradient);
}

const std::tr1::shared_ptr<dotk::DOTk_Primal> & DOTk_OptimizationDataMng::getPrimalStruc() const
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::getPrimalStruc. ABORT. ****\n");
    std::abort();
}

void DOTk_OptimizationDataMng::computeGradient()
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::computeGradient. ABORT. ****\n");
    std::abort();
}

void DOTk_OptimizationDataMng::computeGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
                                                 const std::tr1::shared_ptr<dotk::vector<Real> > & gradient_)
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::computeGradient(in). ABORT. ****\n");
    std::abort();
}

void DOTk_OptimizationDataMng::applyVectorToHessian(const std::tr1::shared_ptr<dotk::vector<Real> > & input_,
                                                    const std::tr1::shared_ptr<dotk::vector<Real> > & output_)
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::applyVectorToHessian(in,out). ABORT. ****\n");
    std::abort();
}

Real DOTk_OptimizationDataMng::evaluateObjective()
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::evaluateObjective. ABORT. ****\n");
    std::abort();
}

Real DOTk_OptimizationDataMng::evaluateObjective(const std::tr1::shared_ptr<dotk::vector<Real> > & input_)
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::evaluateObjective(in). ABORT. ****\n");
    std::abort();
}

size_t DOTk_OptimizationDataMng::getObjectiveFunctionEvaluationCounter() const
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::getObjectiveFuncEvalCounter. ABORT. ****\n");
    std::abort();
}

const std::tr1::shared_ptr<dotk::DOTk_AssemblyManager> & DOTk_OptimizationDataMng::getRoutinesMng() const
{
    std::perror("\n**** Unimplemented Function DOTk_OptimizationDataMng::getRoutinesMng. ABORT. ****\n");
    std::abort();
}

void DOTk_OptimizationDataMng::initialize(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_)
{
    if(primal_->dual().use_count() > 0)
    {
        m_NewDual = primal_->dual()->clone();
        m_OldDual = primal_->dual()->clone();
        m_NewDual->copy(*primal_->dual());
        m_OldDual->copy(*primal_->dual());
    }
    m_NewPrimal->copy(*primal_->control());
}

}
