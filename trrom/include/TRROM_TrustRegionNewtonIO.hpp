/*
 * TRROM_TrustRegionNewtonIO.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_TRUSTREGIONNEWTONIO_HPP_
#define TRROM_TRUSTREGIONNEWTONIO_HPP_

#include <string>
#include <memory>
#include <fstream>

#include "TRROM_Types.hpp"

namespace trrom
{

class TrustRegionStepMng;
class SteihaugTointSolver;
class OptimizationDataMng;

template<typename ScalarType>
class Vector;

class TrustRegionNewtonIO
{
public:
    TrustRegionNewtonIO();
    ~TrustRegionNewtonIO();

    void setNumOptimizationItrDone(int itr_);
    int getNumOptimizationItrDone() const;

    void printOutputPerIteration();
    void printOutputFinalIteration();
    void setDisplayOption(trrom::types::display_t option_);
    trrom::types::display_t getDisplayOption() const;

    void openFile(const std::string & name_);
    void closeFile();

    void printInitialDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_);
    void printSolution(const std::shared_ptr<trrom::Vector<double> > & primal_);
    void printConvergedDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                   const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                   const trrom::TrustRegionStepMng * const step_mng_);
    void printTrustRegionSubProblemDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                               const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                               const trrom::TrustRegionStepMng * const step_mng_);

private:
    void printHeader();
    void printSubProblemDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                    const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                    const trrom::TrustRegionStepMng * const step_mng_);
    void printSubProblemFirstItrDiagnostics(const std::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                            const std::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                            const trrom::TrustRegionStepMng * const step_mng_);
    void printCurrentSolution(const std::shared_ptr<trrom::Vector<double> > & primal_);
    void getSolverExitCriterion(trrom::types::solver_stop_criterion_t type_, std::ostringstream & criterion_);

private:
    int m_NumOptimizationItrDone;
    std::ofstream m_DiagnosticsFile;
    trrom::types::display_t m_DisplayType;

private:
    TrustRegionNewtonIO(const trrom::TrustRegionNewtonIO &);
    trrom::TrustRegionNewtonIO & operator=(const trrom::TrustRegionNewtonIO & rhs_);
};

}

#endif /* TRROM_TRUSTREGIONNEWTONIO_HPP_ */
