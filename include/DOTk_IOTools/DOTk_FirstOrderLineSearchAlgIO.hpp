/*
 * DOTk_FirstOrderLineSearchAlgIO.hpp
 *
 *  Created on: Sep 18, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_FIRSTORDERLINESEARCHALGIO_HPP_
#define DOTK_FIRSTORDERLINESEARCHALGIO_HPP_

namespace dotk
{

class DOTk_LineSearchStepMng;
class DOTk_LineSearchAlgorithmsDataMng;

template<class Type>
class vector;

class DOTk_FirstOrderLineSearchAlgIO
{
public:
    DOTk_FirstOrderLineSearchAlgIO();
    virtual ~DOTk_FirstOrderLineSearchAlgIO();

    void display(dotk::types::display_t input_);
    dotk::types::display_t display() const;

    void license();
    void license(bool input_);

    void closeFile();
    void openFile(const char* const name_);
    void printDiagnosticsReport(const std::tr1::shared_ptr<dotk::DOTk_LineSearchStepMng> & step_,
                                const std::tr1::shared_ptr<dotk::DOTk_LineSearchAlgorithmsDataMng> & mng_);

private:
    bool m_PrintLicenseFlag;
    std::ofstream m_DiagnosticsFile;
    dotk::types::display_t m_DisplayFlag;

private:
    DOTk_FirstOrderLineSearchAlgIO(const dotk::DOTk_FirstOrderLineSearchAlgIO &);
    dotk::DOTk_FirstOrderLineSearchAlgIO operator=(const dotk::DOTk_FirstOrderLineSearchAlgIO & rhs_);
};

}

#endif /* DOTK_FIRSTORDERLINESEARCHALGIO_HPP_ */
