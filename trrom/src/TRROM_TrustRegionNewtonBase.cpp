/*
 * TRROM_TrustRegionNewtonBase.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_BoundConstraints.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_OptimizationDataMng.hpp"
#include "TRROM_TrustRegionNewtonBase.hpp"

namespace trrom
{

TrustRegionNewtonBase::TrustRegionNewtonBase(const std::tr1::shared_ptr<trrom::Data> & data_) :
        m_MaxNumUpdates(10),
        m_MaxNumOptimizationItr(100),
        m_NumOptimizationItrDone(0),
        m_GradientTolerance(1e-10),
        m_TrialStepTolerance(1e-10),
        m_ObjectiveTolerance(1e-10),
        m_StagnationTolerance(1e-12),
        m_StationarityMeasure(0.),
        m_ActualReductionTolerance(1e-10),
        m_StoppingCriterion(trrom::types::OPT_ALG_HAS_NOT_CONVERGED),
        m_WorkVector(data_->control()->create()),
        m_LowerBound(data_->getControlLowerBound()),
        m_UpperBound(data_->getControlUpperBound()),
        m_BoundConstraint(new trrom::BoundConstraints)
{
}

TrustRegionNewtonBase::~TrustRegionNewtonBase()
{
}

void TrustRegionNewtonBase::setGradientTolerance(double input_)
{
    m_GradientTolerance = input_;
}

double TrustRegionNewtonBase::getGradientTolerance() const
{
    return (m_GradientTolerance);
}

void TrustRegionNewtonBase::setTrialStepTolerance(double input_)
{
    m_TrialStepTolerance = input_;
}

double TrustRegionNewtonBase::getTrialStepTolerance() const
{
    return (m_TrialStepTolerance);
}

void TrustRegionNewtonBase::setObjectiveTolerance(double input_)
{
    m_ObjectiveTolerance = input_;
}

double TrustRegionNewtonBase::getObjectiveTolerance() const
{
    return (m_ObjectiveTolerance);
}

void TrustRegionNewtonBase::setStagnationTolerance(double input_)
{
    m_StagnationTolerance = input_;
}

double TrustRegionNewtonBase::getStagnationTolerance() const
{
    return (m_StagnationTolerance);
}

void TrustRegionNewtonBase::setActualReductionTolerance(double input_)
{
    m_ActualReductionTolerance = input_;
}

double TrustRegionNewtonBase::getActualReductionTolerance() const
{
    return (m_ActualReductionTolerance);
}

double TrustRegionNewtonBase::getStationarityMeasure() const
{
    return (m_StationarityMeasure);
}

void TrustRegionNewtonBase::setMaxNumUpdates(int input_)
{
    m_MaxNumUpdates = input_;
}

int TrustRegionNewtonBase::getMaxNumUpdates() const
{
    return (m_MaxNumUpdates);
}

void TrustRegionNewtonBase::setNumOptimizationItrDone(int input_)
{
    m_NumOptimizationItrDone = input_;
}

int TrustRegionNewtonBase::getNumOptimizationItrDone() const
{
    return (m_NumOptimizationItrDone);
}

void TrustRegionNewtonBase::setMaxNumOptimizationItr(int input_)
{
    m_MaxNumOptimizationItr = input_;
}

int TrustRegionNewtonBase::getMaxNumOptimizationItr() const
{
    return (m_MaxNumOptimizationItr);
}

void TrustRegionNewtonBase::setStoppingCriterion(trrom::types::stop_criterion_t input_)
{
    m_StoppingCriterion = input_;
}

trrom::types::stop_criterion_t TrustRegionNewtonBase::getStoppingCriterion() const
{
    return (m_StoppingCriterion);
}

bool TrustRegionNewtonBase::updatePrimal(const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_,
                                          const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_,
                                          const std::tr1::shared_ptr<trrom::Vector<double> > & mid_gradient_)
{
    bool primal_updated = false;

    double xi = 1.;
    double beta = 1e-2;
    double alpha = beta;
    double mu = 1. - 1e-4;
    double mid_actual_reduction = step_->getActualReduction();
    double mid_objective_value = step_->getMidObejectiveFunctionValue();

    int iteration = 0;
    int max_num_updates = this->getMaxNumUpdates();
    while(iteration < max_num_updates)
    {
        // Compute trial point based on the mid gradient (i.e. mid steepest descent)
        double lambda = -xi / alpha;
        m_WorkVector->update(1., *step_->getMidPrimal(), 0.);
        m_WorkVector->update(lambda, *mid_gradient_, 1.);
        m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_WorkVector);
        // Compute trial objective function
        double trial_objective_value = data_->evaluateObjective(m_WorkVector);
        // Compute actual reduction
        double trial_actual_reduction = trial_objective_value - mid_objective_value;
        // Check convergence
        if(trial_actual_reduction < -mu * mid_actual_reduction)
        {
            primal_updated = true;
            data_->getNewPrimal()->update(1., *m_WorkVector, 0.);
            step_->setActualReduction(trial_actual_reduction);
            data_->setNewObjectiveFunctionValue(trial_objective_value);
            break;
        }
        // Compute scaling for next iteration
        if(iteration == 1)
        {
            xi = alpha;
        }
        else
        {
            xi = xi * beta;
        }
        ++iteration;
    }

    if(iteration >= max_num_updates)
    {
        data_->getNewPrimal()->update(1., *step_->getMidPrimal(), 0.);
        data_->setNewObjectiveFunctionValue(mid_objective_value);
    }

    return (primal_updated);
}

void TrustRegionNewtonBase::updateDataManager(const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_,
                                               const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_,
                                               const std::tr1::shared_ptr<trrom::Vector<double> > & mid_gradient_,
                                               const std::tr1::shared_ptr<trrom::Vector<double> > & inactive_set_)
{
    // set new objective function value
    double current_objective_value = data_->getNewObjectiveFunctionValue();
    data_->setOldObjectiveFunctionValue(current_objective_value);
    // update primal vector
    data_->getOldPrimal()->update(1., *data_->getNewPrimal(), 0.);
    data_->getOldGradient()->update(1., *data_->getNewGradient(), 0.);

    if(this->updatePrimal(step_, data_, mid_gradient_) == true)
    {
        // update new gradient since primal was successfully updated;
        // else, keep mid gradient and thus mid primal
        data_->computeGradient();
    }
    else
    {
        data_->getNewGradient()->update(1., *mid_gradient_, 0.);
    }

    // compute norm of projected gradient
    m_WorkVector->update(1., *data_->getNewGradient(), 0.);
    m_WorkVector->elementWiseMultiplication(*inactive_set_);
    double norm_proj_gradient = m_WorkVector->norm();
    data_->setNormNewGradient(norm_proj_gradient);
    // compute gradient inexactness bound
    step_->updateGradientInexactnessTolerance(norm_proj_gradient);
    double gradient_inexactness_tolerance = data_->getGradientInexactnessTolerance();
    data_->setGradientInexactnessTolerance(gradient_inexactness_tolerance);
}

bool TrustRegionNewtonBase::checkStoppingCriteria(const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_,
                                                   const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_)
{
    double actual_reduction = step_->getActualReduction();
    double norm_proj_gradient = data_->getNormNewGradient();
    double objective_function_value = data_->getNewObjectiveFunctionValue();

    bool optimization_algorithm_converged = false;
    if(m_StationarityMeasure <= this->getTrialStepTolerance())
    {
        optimization_algorithm_converged = true;
        this->setStoppingCriterion(trrom::types::TRIAL_STEP_TOL_SATISFIED);
    }
    else if(std::isfinite(m_StationarityMeasure) == false)
    {
        optimization_algorithm_converged = true;
        this->resetCurrentStateToPreviousState(data_);
        this->setStoppingCriterion(trrom::types::NaN_TRIAL_STEP_NORM);
    }
    else if(norm_proj_gradient < this->getGradientTolerance())
    {
        optimization_algorithm_converged = true;
        this->setStoppingCriterion(trrom::types::GRADIENT_TOL_SATISFIED);
    }
    else if(std::isfinite(norm_proj_gradient) == false)
    {
        optimization_algorithm_converged = true;
        this->resetCurrentStateToPreviousState(data_);
        this->setStoppingCriterion(trrom::types::NaN_GRADIENT_NORM);
    }
    else if(std::abs(actual_reduction) <= this->getActualReductionTolerance())
    {
        // objective function stagnation
        optimization_algorithm_converged = true;
        this->setStoppingCriterion(trrom::types::ACTUAL_REDUCTION_TOL_SATISFIED);
    }
    else if(objective_function_value <= this->getObjectiveTolerance())
    {
        // objective function stagnation
        optimization_algorithm_converged = true;
        this->setStoppingCriterion(trrom::types::OBJECTIVE_FUNC_TOL_SATISFIED);
    }
    else if(this->getNumOptimizationItrDone() >= this->getMaxNumOptimizationItr())
    {
        optimization_algorithm_converged = true;
        this->setStoppingCriterion(trrom::types::MAX_NUM_ITR_REACHED);
    }

    return (optimization_algorithm_converged);
}

void TrustRegionNewtonBase::computeStationarityMeasure(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_,
                                                        const std::tr1::shared_ptr<trrom::Vector<double> > & inactive_set_)
{
    m_WorkVector->update(1., *data_->getNewPrimal(), 0.);
    m_WorkVector->update(-1., *data_->getNewGradient(), 1.);
    m_BoundConstraint->project(*m_LowerBound, *m_UpperBound, *m_WorkVector);
    m_WorkVector->scale(-1.);
    m_WorkVector->update(1., *data_->getNewPrimal(), 1.);
    m_WorkVector->elementWiseMultiplication(*inactive_set_);

    m_StationarityMeasure = m_WorkVector->norm();
    data_->setNormTrialStep(m_StationarityMeasure);
}

void TrustRegionNewtonBase::resetCurrentStateToPreviousState(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_)
{
    data_->getNewPrimal()->update(1., *data_->getOldPrimal(), 0.);
    data_->getNewGradient()->update(1., *data_->getOldGradient(), 0.);
    data_->setNewObjectiveFunctionValue(data_->getOldObjectiveFunctionValue());
}

}
