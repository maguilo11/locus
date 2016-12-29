/*
 * DOTk_MexDiagnostics.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexDiagnostics.hpp"
#include "DOTk_MexAlgorithmParser.hpp"

namespace dotk
{

DOTk_MexDiagnostics::DOTk_MexDiagnostics(const mxArray* options_) :
        m_LowerSuperScrips(0),
        m_UpperSuperScrips(0),
        m_ProblemType(dotk::types::PROBLEM_TYPE_UNDEFINED)
{
    this->initialize(options_);
}

DOTk_MexDiagnostics::~DOTk_MexDiagnostics()
{
}

int DOTk_MexDiagnostics::getLowerSuperScript() const
{
    return (m_LowerSuperScrips);
}

int DOTk_MexDiagnostics::getUpperSuperScript() const
{
    return (m_UpperSuperScrips);
}

dotk::types::problem_t DOTk_MexDiagnostics::getProblemType() const
{
    return (m_ProblemType);
}

void DOTk_MexDiagnostics::initialize(const mxArray* options_)
{
    m_ProblemType = dotk::mex::parseProblemType(options_);
    m_LowerSuperScrips = dotk::mex::parseFiniteDifferenceDiagnosticsLowerSuperScripts(options_);
    m_UpperSuperScrips = dotk::mex::parseFiniteDifferenceDiagnosticsUpperSuperScripts(options_);
}

}
