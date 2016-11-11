/*
 * TRROM_OptimizationDataMng.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_OptimizationDataMng.hpp"

namespace trrom
{

OptimizationDataMng::OptimizationDataMng() :
        m_IterationCounter(0),
        m_NormTrialStep(0),
        m_NormNewGradient(0),
        m_StagnationMeasure(0),
        m_OldObjectiveFunction(0),
        m_NewObjectiveFunction(0),
        m_GradientInexactnessTolerance(0),
        m_ObjectiveInexactnessTolerance(0),
        m_GradientInexactnessToleranceExceeded(false),
        m_ObjectiveInexactnessToleranceExceeded(false),
        m_TrialStep(),
        m_OldPrimal(),
        m_NewPrimal(),
        m_OldGradient(),
        m_NewGradient(),
        m_MatrixTimesVector()
{
}

OptimizationDataMng::OptimizationDataMng(const std::tr1::shared_ptr<trrom::Data> & data_) :
        m_IterationCounter(0),
        m_NormTrialStep(0),
        m_NormNewGradient(0),
        m_StagnationMeasure(0),
        m_OldObjectiveFunction(0),
        m_NewObjectiveFunction(0),
        m_GradientInexactnessTolerance(0),
        m_ObjectiveInexactnessTolerance(0),
        m_GradientInexactnessToleranceExceeded(false),
        m_ObjectiveInexactnessToleranceExceeded(false),
        m_TrialStep(data_->control()->create()),
        m_OldPrimal(data_->control()->create()),
        m_NewPrimal(data_->control()->create()),
        m_OldGradient(data_->control()->create()),
        m_NewGradient(data_->control()->create()),
        m_MatrixTimesVector(data_->control()->create())
{
    this->initialize(data_);
}

OptimizationDataMng::~OptimizationDataMng()
{
}

int OptimizationDataMng::getIterationCounter() const
{
    return (m_IterationCounter);
}

void OptimizationDataMng::setIterationCounter(int input_)
{
    m_IterationCounter = input_;
}

void OptimizationDataMng::setNormTrialStep(double input_)
{
    m_NormTrialStep = input_;
}

double OptimizationDataMng::getNormTrialStep() const
{
    return (m_NormTrialStep);
}

void OptimizationDataMng::setNormNewGradient(double input_)
{
    m_NormNewGradient = input_;
}

double OptimizationDataMng::getNormNewGradient() const
{
    return (m_NormNewGradient);
}

void OptimizationDataMng::setStagnationMeasure(double input_)
{
    m_StagnationMeasure = input_;
}

double OptimizationDataMng::getStagnationMeasure() const
{
    return (m_StagnationMeasure);
}

void OptimizationDataMng::setGradientInexactnessFlag(bool input_)
{
    m_GradientInexactnessToleranceExceeded = input_;
}

bool OptimizationDataMng::isGradientInexactnessToleranceExceeded()
{
    return (m_GradientInexactnessToleranceExceeded);
}

void OptimizationDataMng::setGradientInexactnessTolerance(double input_)
{
    m_GradientInexactnessTolerance = input_;
}

double OptimizationDataMng::getGradientInexactnessTolerance() const
{
    return (m_GradientInexactnessTolerance);
}

void OptimizationDataMng::setObjectiveInexactnessFlag(bool input_)
{
    m_ObjectiveInexactnessToleranceExceeded = input_;
}

bool OptimizationDataMng::isObjectiveInexactnessToleranceExceeded()
{
    return (m_ObjectiveInexactnessToleranceExceeded);
}

void OptimizationDataMng::setObjectiveInexactnessTolerance(double input_)
{
    m_ObjectiveInexactnessTolerance = input_;
}

double OptimizationDataMng::getObjectiveInexactnessTolerance() const
{
    return (m_ObjectiveInexactnessTolerance);
}

void OptimizationDataMng::setNewObjectiveFunctionValue(double value_)
{
    m_NewObjectiveFunction = value_;
}

double OptimizationDataMng::getNewObjectiveFunctionValue() const
{
    return (m_NewObjectiveFunction);
}

void OptimizationDataMng::setOldObjectiveFunctionValue(double value_)
{
    m_OldObjectiveFunction = value_;
}

double OptimizationDataMng::getOldObjectiveFunctionValue() const
{
    return (m_OldObjectiveFunction);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & OptimizationDataMng::getMatrixTimesVector() const
{
    return (m_MatrixTimesVector);
}

void OptimizationDataMng::setTrialStep(const trrom::Vector<double> & input_)
{
    m_TrialStep->copy(input_);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & OptimizationDataMng::getTrialStep() const
{
    return (m_TrialStep);
}

void OptimizationDataMng::setNewPrimal(const trrom::Vector<double> & input_)
{
    m_NewPrimal->copy(input_);
}

void OptimizationDataMng::setOldPrimal(const trrom::Vector<double> & input_)
{
    m_OldPrimal->copy(input_);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & OptimizationDataMng::getNewPrimal() const
{
    return (m_NewPrimal);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & OptimizationDataMng::getOldPrimal() const
{
    return (m_OldPrimal);
}

void OptimizationDataMng::setNewGradient(const trrom::Vector<double> & input_)
{
    m_NewGradient->copy(input_);
}

void OptimizationDataMng::setOldGradient(const trrom::Vector<double> & input_)
{
    m_OldGradient->copy(input_);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & OptimizationDataMng::getNewGradient() const
{
    return (m_NewGradient);
}

const std::tr1::shared_ptr<trrom::Vector<double> > & OptimizationDataMng::getOldGradient() const
{
    return (m_OldGradient);
}

void OptimizationDataMng::initialize(const std::tr1::shared_ptr<trrom::Data> & data_)
{
    m_NewPrimal->copy(*data_->control());
}

}
