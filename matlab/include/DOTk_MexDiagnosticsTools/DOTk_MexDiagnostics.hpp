/*
 * DOTk_MexDiagnostics.hpp
 *
 *  Created on: May 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXDIAGNOSTICS_HPP_
#define DOTK_MEXDIAGNOSTICS_HPP_

#include <mex.h>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_MexDiagnostics
{
public:
    explicit DOTk_MexDiagnostics(const mxArray* options_);
    virtual ~DOTk_MexDiagnostics();

    int getLowerSuperScript() const;
    int getUpperSuperScript() const;
    dotk::types::problem_t getProblemType() const;

    virtual void checkFirstDerivative(const mxArray* input_[]) = 0;
    virtual void checkSecondDerivative(const mxArray* input_[]) = 0;

private:
    void initialize(const mxArray* options_);

private:
    int m_LowerSuperScrips;
    int m_UpperSuperScrips;
    dotk::types::problem_t m_ProblemType;

private:
    DOTk_MexDiagnostics(const dotk::DOTk_MexDiagnostics&);
    dotk::DOTk_MexDiagnostics& operator=(const dotk::DOTk_MexDiagnostics&);
};

}

#endif /* DOTK_MEXDIAGNOSTICS_HPP_ */
