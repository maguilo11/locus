/*
 * DOTk_MexDiagnosticsTypeNLP.hpp
 *
 *  Created on: May 5, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXDIAGNOSTICSTYPENLP_HPP_
#define DOTK_MEXDIAGNOSTICSTYPENLP_HPP_

#include "DOTk_MexDiagnostics.hpp"

namespace dotk
{

class DOTk_MexDiagnosticsTypeNLP : public dotk::DOTk_MexDiagnostics
{
public:
    explicit DOTk_MexDiagnosticsTypeNLP(const mxArray* input_[]);
    ~DOTk_MexDiagnosticsTypeNLP();

    void checkFirstDerivative(const mxArray* input_[]);
    void checkSecondDerivative(const mxArray* input_[]);

private:
    void checkFirstDerivativeTypeUNLP(const mxArray* input_[]);
    void checkFirstDerivativeTypeENLP(const mxArray* input_[]);
    void checkFirstDerivativeTypeCNLP(const mxArray* input_[]);
    void checkSecondDerivativeTypeUNLP(const mxArray* input_[]);
    void checkSecondDerivativeTypeENLP(const mxArray* input_[]);
    void checkSecondDerivativeTypeCNLP(const mxArray* input_[]);

    void checkObjectiveFunctionFirstDerivative(const mxArray* input_[]);
    void checkObjectiveFunctionSecondDerivative(const mxArray* input_[]);
    void checkEqualityConstraintFirstDerivative(const mxArray* input_[]);
    void checkEqualityConstraintSecondDerivative(const mxArray* input_[]);
    void checkInequalityConstraintFirstDerivative(const mxArray* input_[]);
    void checkInequalityConstraintSecondDerivative(const mxArray* input_[]);

private:
    DOTk_MexDiagnosticsTypeNLP(const dotk::DOTk_MexDiagnosticsTypeNLP&);
    dotk::DOTk_MexDiagnosticsTypeNLP& operator=(const dotk::DOTk_MexDiagnosticsTypeNLP&);
};

}

#endif /* DOTK_MEXDIAGNOSTICSTYPENLP_HPP_ */
