/*
 * TRROM_KelleySachs.cpp
 *
 *  Created on: Sep 5, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <cmath>
#include <string>

#include "TRROM_Data.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_KelleySachs.hpp"
#include "TRROM_KelleySachsStepMng.hpp"
#include "TRROM_SteihaugTointDataMng.hpp"
#include "TRROM_SteihaugTointNewtonIO.hpp"
#include "TRROM_ProjectedSteihaugTointPcg.hpp"

namespace trrom
{

KelleySachs::KelleySachs(const std::tr1::shared_ptr<trrom::Data> & data_,
                         const std::tr1::shared_ptr<trrom::SteihaugTointDataMng> & data_mng_,
                         const std::tr1::shared_ptr<trrom::KelleySachsStepMng> & step_mng) :
        trrom::TrustRegionKelleySachs(data_),
        m_MidGradient(data_->control()->create()),
        m_IO(new trrom::SteihaugTointNewtonIO),
        m_StepMng(step_mng),
        m_DataMng(data_mng_),
        m_Solver(new trrom::ProjectedSteihaugTointPcg(data_))
{
}

KelleySachs::~KelleySachs()
{
}

void KelleySachs::setMaxNumSolverItr(int input_)
{
    m_Solver->setMaxNumItr(input_);
}

void KelleySachs::getMin()
{
    std::string name("KelleySachsDiagnostics.out");
    m_IO->openFile(name);

    double new_objective_value = m_DataMng->evaluateObjective();
    m_DataMng->setNewObjectiveFunctionValue(new_objective_value);
    m_DataMng->computeGradient();
    m_DataMng->getOldPrimal()->copy(*m_DataMng->getNewPrimal());
    m_DataMng->getOldGradient()->copy(*m_DataMng->getNewGradient());
    double norm_gradient = m_DataMng->getNewGradient()->norm();
    m_DataMng->setNormNewGradient(norm_gradient);
    if(m_StepMng->isInitialTrustRegionRadiusSetToGradNorm() == true)
    {
        m_StepMng->setTrustRegionRadius(norm_gradient);
    }
    trrom::TrustRegionKelleySachs::computeStationarityMeasure(m_DataMng, m_Solver->getInactiveSet());

    m_IO->printInitialDiagnostics(m_DataMng);

    int itr = 1;
    while(1)
    {
        this->updateNumOptimizationItrDone(itr);
        // Compute adaptive constants to ensure superlinear convergence
        double measure = trrom::TrustRegionKelleySachs::getStationarityMeasure();
        double value = std::pow(measure, static_cast<double>(0.75));
        double epsilon = std::min(static_cast<double>(1e-3), value);
        m_StepMng->setEpsilon(epsilon);
        value = std::pow(measure, static_cast<double>(0.95));
        double eta = static_cast<double>(0.1) * std::min(static_cast<double>(1e-1), value);
        m_StepMng->setEta(eta);
        // Solve trust region subproblem
        m_StepMng->solveSubProblem(m_DataMng, m_Solver, m_IO);
        // Compute new midpoint gradient
        m_DataMng->computeGradient(m_StepMng->getMidPrimal(), m_MidGradient);
        // Update current primal and gradient information
        trrom::TrustRegionKelleySachs::updateDataManager(m_StepMng,
                                                         m_DataMng,
                                                         m_MidGradient,
                                                         m_Solver->getInactiveSet());
        if(trrom::TrustRegionKelleySachs::checkStoppingCriteria(m_StepMng, m_DataMng) == true)
        {
            m_IO->printConvergedDiagnostics(m_DataMng, m_Solver, m_StepMng.get());
            break;
        }
        // Update stationarity measure
        trrom::TrustRegionKelleySachs::computeStationarityMeasure(m_DataMng, m_Solver->getInactiveSet());
        ++itr;
    }

    m_IO->printSolution(m_DataMng->getNewPrimal());
    m_IO->closeFile();
}

void KelleySachs::printDiagnostics()
{
    m_IO->setDisplayOption(trrom::types::FINAL);
}

void KelleySachs::updateNumOptimizationItrDone(const int & input_)
{
    m_IO->setNumOptimizationItrDone(input_);
    trrom::TrustRegionKelleySachs::setNumOptimizationItrDone(input_);
}

}
