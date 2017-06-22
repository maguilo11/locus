/*
 * DOTk_LineSearchInexactNewtonIO.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINESEARCHINEXACTNEWTONIO_HPP_
#define DOTK_LINESEARCHINEXACTNEWTONIO_HPP_

#include <memory>
#include <fstream>

namespace dotk
{

class DOTk_KrylovSolver;
class DOTk_LineSearchStepMng;
class DOTk_LineSearchAlgorithmsDataMng;

class DOTk_LineSearchInexactNewtonIO
{
public:
    DOTk_LineSearchInexactNewtonIO();
    virtual ~DOTk_LineSearchInexactNewtonIO();

    void display(dotk::types::display_t input_);
    dotk::types::display_t display() const;

    void license();
    void license(bool input_);

    void closeFile();
    void openFile(const char * const name_);
    void printDiagnosticsReport(const std::shared_ptr<dotk::DOTk_KrylovSolver> & solver_,
                                const std::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                const std::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_);

private:
    bool m_PrintLicenseFlag;
    std::ofstream m_DiagnosticsFile;
    dotk::types::display_t m_DisplayFlag;

private:
    DOTk_LineSearchInexactNewtonIO(const dotk::DOTk_LineSearchInexactNewtonIO &);
    dotk::DOTk_LineSearchInexactNewtonIO operator=(const dotk::DOTk_LineSearchInexactNewtonIO & rhs_);
};

}

#endif /* DOTK_LINESEARCHINEXACTNEWTONIO_HPP_ */
