/*
 * DOTk_InexactTrustRegionSqpIO.hpp
 *
 *  Created on: Feb 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_INEXACTTRUSTREGIONSQPIO_HPP_
#define DOTK_INEXACTTRUSTREGIONSQPIO_HPP_

#include <fstream>

namespace dotk
{

class DOTk_InexactTrustRegionSQP;
class DOTk_TrustRegionMngTypeELP;
class DOTk_InexactTrustRegionSqpSolverMng;

class DOTk_InexactTrustRegionSqpIO
{
public:
    DOTk_InexactTrustRegionSqpIO();
    virtual ~DOTk_InexactTrustRegionSqpIO();

    void display(dotk::types::display_t input_);
    dotk::types::display_t display() const;

    void license();
    void license(bool input_);

    void closeFile();
    void openFile(const char * const name_);
    void printDiagnosticsReport(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_,
                                const std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_mng_);

private:
    void writeHeader();
    void writeInitialDiagnostics(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                 const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_);
    void writeFirstTrustRegionSubProblemItrDiagnostics(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_,
                                                       const std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_);
    void writeTrustRegionSubProblemDiagnostics(const dotk::DOTk_InexactTrustRegionSQP* const alg_,
                                               const std::tr1::shared_ptr<dotk::DOTk_TrustRegionMngTypeELP> & mng_,
                                               const std::tr1::shared_ptr<dotk::DOTk_InexactTrustRegionSqpSolverMng> & solver_);

private:
    bool m_PrintLicenseFlag;
    std::ofstream m_DiagnosticsFile;
    dotk::types::display_t m_DisplayFlag;

private:
    DOTk_InexactTrustRegionSqpIO(const dotk::DOTk_InexactTrustRegionSqpIO&);
    dotk::DOTk_InexactTrustRegionSqpIO operator=(const dotk::DOTk_InexactTrustRegionSqpIO&);
};

}

#endif /* DOTK_INEXACTTRUSTREGIONSQPIO_HPP_ */
