/*
 * TRROM_SteihaugTointNewtonIO.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_STEIHAUGTOINTNEWTONIO_HPP_
#define TRROM_STEIHAUGTOINTNEWTONIO_HPP_

#include <string>
#include <fstream>

#include "TRROM_Types.hpp"

namespace trrom
{

class TrustRegionStepMng;
class SteihaugTointSolver;
class OptimizationDataMng;

template<typename ScalarType>
class Vector;

class SteihaugTointNewtonIO
{
public:
    SteihaugTointNewtonIO();
    ~SteihaugTointNewtonIO();

    void setNumOptimizationItrDone(int itr_);
    int getNumOptimizationItrDone() const;

    void printOutputPerIteration();
    void printOutputFinalIteration();
    void setDisplayOption(trrom::types::display_t option_);
    trrom::types::display_t getDisplayOption() const;

    void openFile(const std::string & name_);
    void closeFile();

    void printInitialDiagnostics(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_mng_);
    void printSolution(const std::tr1::shared_ptr<trrom::Vector<double> > & primal_);
    void printConvergedDiagnostics(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                   const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                   const trrom::TrustRegionStepMng * const step_mng_);
    void printTrustRegionSubProblemDiagnostics(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                               const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                               const trrom::TrustRegionStepMng * const step_mng_);

private:
    void printHeader();
    void printSubProblemDiagnostics(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                    const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                    const trrom::TrustRegionStepMng * const step_mng_);
    void printSubProblemFirstItrDiagnostics(const std::tr1::shared_ptr<trrom::OptimizationDataMng> & data_mng_,
                                            const std::tr1::shared_ptr<trrom::SteihaugTointSolver> & solver_,
                                            const trrom::TrustRegionStepMng * const step_mng_);
    void printCurrentSolution(const std::tr1::shared_ptr<trrom::Vector<double> > & primal_);
    void getSolverExitCriterion(trrom::types::solver_stop_criterion_t type_, std::ostringstream & criterion_);

private:
    int m_NumOptimizationItrDone;
    std::ofstream m_DiagnosticsFile;
    trrom::types::display_t m_DisplayType;

private:
    SteihaugTointNewtonIO(const trrom::SteihaugTointNewtonIO &);
    trrom::SteihaugTointNewtonIO & operator=(const trrom::SteihaugTointNewtonIO & rhs_);
};

}

#endif /* TRROM_STEIHAUGTOINTNEWTONIO_HPP_ */
