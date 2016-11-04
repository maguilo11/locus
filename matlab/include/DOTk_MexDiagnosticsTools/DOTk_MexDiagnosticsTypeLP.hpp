/*
 * DOTk_MexDiagnosticsTypeLP.hpp
 *
 *  Created on: May 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXDIAGNOSTICSTYPELP_HPP_
#define DOTK_MEXDIAGNOSTICSTYPELP_HPP_

#include "DOTk_MexDiagnostics.hpp"

namespace dotk
{

class DOTk_MexDiagnosticsTypeLP : public dotk::DOTk_MexDiagnostics
{
public:
    explicit DOTk_MexDiagnosticsTypeLP(const mxArray* input_[]);
    ~DOTk_MexDiagnosticsTypeLP();

    void checkFirstDerivative(const mxArray* input_[]);
    void checkSecondDerivative(const mxArray* input_[]);

private:
    void checkFirstDerivativeTypeULP(const mxArray* input_[]);
    void checkFirstDerivativeTypeELP(const mxArray* input_[]);
    void checkFirstDerivativeTypeCLP(const mxArray* input_[]);
    void checkFirstDerivativeTypeILP(const mxArray* input_[]);
    void checkSecondDerivativeTypeULP(const mxArray* input_[]);
    void checkSecondDerivativeTypeELP(const mxArray* input_[]);
    void checkSecondDerivativeTypeCLP(const mxArray* input_[]);

private:
    DOTk_MexDiagnosticsTypeLP(const dotk::DOTk_MexDiagnosticsTypeLP&);
    dotk::DOTk_MexDiagnosticsTypeLP& operator=(const dotk::DOTk_MexDiagnosticsTypeLP&);
};

}

#endif /* DOTK_MEXDIAGNOSTICSTYPELP_HPP_ */
