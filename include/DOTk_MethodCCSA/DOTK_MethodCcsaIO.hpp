/*
 * DOTK_MethodCcsaIO.hpp
 *
 *  Created on: Dec 16, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_METHODCCSAIO_HPP_
#define DOTK_METHODCCSAIO_HPP_

#include <fstream>
#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_DataMngCCSA;
class DOTk_AlgorithmCCSA;

class DOTK_MethodCcsaIO
{
public:
    DOTK_MethodCcsaIO();
    ~DOTK_MethodCcsaIO();

    dotk::types::display_t getDisplayOption() const;

    void printSolutionAtEachIteration();
    void printSolutionAtFinalIteration();
    void openFile(const char * const name_);
    void closeFile();

    void printSolution(const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);
    void print(const dotk::DOTk_AlgorithmCCSA* const algorithm_,
               const std::tr1::shared_ptr<dotk::DOTk_DataMngCCSA> & data_mng_);

private:
    void printHeader();

private:
    std::ofstream m_OutputFileStream;
    dotk::types::display_t m_DisplayType;

private:
    DOTK_MethodCcsaIO(const dotk::DOTK_MethodCcsaIO &);
    dotk::DOTK_MethodCcsaIO & operator=(const dotk::DOTK_MethodCcsaIO & rhs_);
};

}

#endif /* DOTK_METHODCCSAIO_HPP_ */
