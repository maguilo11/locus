/*
 * DOTk_SteihaugTointNewtonIO.hpp
 *
 *  Created on: Sep 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STEIHAUGTOINTNEWTONIO_HPP_
#define DOTK_STEIHAUGTOINTNEWTONIO_HPP_

#include <fstream>
#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_TrustRegionStepMng;
class DOTk_SteihaugTointSolver;
class DOTk_OptimizationDataMng;

template<typename Type>
class vector;

class DOTk_SteihaugTointNewtonIO
{
public:
    DOTk_SteihaugTointNewtonIO();
    ~DOTk_SteihaugTointNewtonIO();

    void setNumOptimizationItrDone(size_t itr_);
    size_t getNumOptimizationItrDone() const;

    void printOutputPerIteration();
    void printOutputFinalIteration();
    void setDisplayOption(dotk::types::display_t option_);
    dotk::types::display_t getDisplayOption() const;

    void openFile(const char * const name_);
    void closeFile();

    void printInitialDiagnostics(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_);
    void printSolution(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_);
    void printConvergedDiagnostics(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                   const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                   const dotk::DOTk_TrustRegionStepMng * const step_mng_);
    void printTrustRegionSubProblemDiagnostics(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                               const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                               const dotk::DOTk_TrustRegionStepMng * const step_mng_);

private:
    void printHeader();
    void printSubProblemDiagnostics(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                    const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                    const dotk::DOTk_TrustRegionStepMng * const step_mng_);
    void printSubProblemFirstItrDiagnostics(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & data_mng_,
                                            const std::tr1::shared_ptr<dotk::DOTk_SteihaugTointSolver> & solver_,
                                            const dotk::DOTk_TrustRegionStepMng * const step_mng_);
    void printCurrentSolution(const std::tr1::shared_ptr<dotk::vector<Real> > & primal_);

private:
    size_t m_NumOptimizationItrDone;
    std::ofstream m_DiagnosticsFile;
    dotk::types::display_t m_DisplayType;

private:
    DOTk_SteihaugTointNewtonIO(const dotk::DOTk_SteihaugTointNewtonIO &);
    dotk::DOTk_SteihaugTointNewtonIO & operator=(const dotk::DOTk_SteihaugTointNewtonIO & rhs_);
};

}

#endif /* DOTK_STEIHAUGTOINTNEWTONIO_HPP_ */