/*
 * DOTk_TrustRegionInexactNewtonIO.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_TRUSTREGIONINEXACTNEWTONIO_HPP_
#define DOTK_TRUSTREGIONINEXACTNEWTONIO_HPP_

#include <memory>
#include <fstream>

namespace dotk
{

class DOTk_KrylovSolver;
class DOTk_TrustRegionInexactNewton;
class DOTk_TrustRegionAlgorithmsDataMng;

class DOTk_TrustRegionInexactNewtonIO
{
public:
    DOTk_TrustRegionInexactNewtonIO();
    virtual ~DOTk_TrustRegionInexactNewtonIO();

    void display(dotk::types::display_t input_);
    dotk::types::display_t display() const;

    void license();
    void license(bool input_);

    void closeFile();
    void openFile(const char * const name_);
    void printDiagnosticReport(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
                               const std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                               const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & opt_mng_,
                               bool did_trust_region_subproblem_converged_ = false);

private:
    bool didTrustRegionSubProblemConverged() const;
    void setTrustRegionSubProblemConvergedFlag(bool input_);

    void writeHeader();
    void writeDiagnostics(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
                          const std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                          const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_);
    void writeInitialDiagnostics(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
                                 const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_);
    void writeTrustRegionSubProblemDiagnostics(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
                                               const std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                                               const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_);
    void writeFullTrustRegionSubProblemDiagnostics(const dotk::DOTk_TrustRegionInexactNewton * const alg_,
                                                   const std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                                                   const std::shared_ptr<dotk::DOTk_TrustRegionAlgorithmsDataMng> & mng_);

private:
    bool m_PrintLicenseFlag;
    std::ofstream m_DiagnosticsFile;
    dotk::types::display_t m_DisplayFlag;
    bool m_TrustRegionSubProblemConverged;

private:
    DOTk_TrustRegionInexactNewtonIO(const dotk::DOTk_TrustRegionInexactNewtonIO &);
    dotk::DOTk_TrustRegionInexactNewtonIO operator=(const dotk::DOTk_TrustRegionInexactNewtonIO & rhs_);
};

}

#endif /* DOTK_TRUSTREGIONINEXACTNEWTONIO_HPP_ */
