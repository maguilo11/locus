/*
 * DOTk_MexAlgorithmParser.cpp
 *
 *  Created on: Apr 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>

#include "vector.hpp"
#include "DOTk_MexApiUtilities.hpp"
#include "DOTk_MexAlgorithmParser.hpp"

namespace dotk
{

namespace mex
{

size_t parseNumberDuals(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberDuals") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberDuals keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberDuals"));
    size_t output = static_cast<size_t>(mxGetScalar(value));
    mxDestroyArray(value);
    return (output);
}

size_t parseNumberStates(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberStates") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberStates keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberStates"));
    size_t output = static_cast<size_t>(mxGetScalar(value));
    mxDestroyArray(value);
    return (output);
}

size_t parseMaxNumUpdates(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MaxNumberUpdates") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MaxNumberUpdates keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberUpdates"));
    size_t output = static_cast<size_t>(mxGetScalar(value));
    mxDestroyArray(value);
    return (output);
}

size_t parseNumberControls(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberControls") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberControls keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberControls"));
    size_t output = static_cast<size_t>(mxGetScalar(value));
    mxDestroyArray(value);
    return (output);
}

size_t parseMaxNumFeasibleItr(const mxArray* input_)
{
    size_t output = 0;
    if(mxGetField(input_, 0, "MaxNumFeasibleItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumFeasibleItr keyword is NULL. MaxNumFeasibleItr set to 5.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 5;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumFeasibleItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseMaxNumOuterIterations(const mxArray* input_)
{
    size_t output = 0;
    if(mxGetField(input_, 0, "MaxNumOuterIterations") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumOuterIterations keyword is NULL. MaxNumOuterIterations set to 100.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 100;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumOuterIterations"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseMaxNumLineSearchItr(const mxArray* input_)
{
    size_t output = 0;
    if(mxGetField(input_, 0, "MaxNumLineSearchItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumLineSearchItr keyword is NULL. MaxNumLineSearchItr set to 10.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 10;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumLineSearchItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseMaxNumTrustRegionSubProblemItr(const mxArray* input_)
{
    size_t output = 0;
    if(mxGetField(input_, 0, "MaxNumTrustRegionSubProblemItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumTrustRegionSubProblemItr keyword is NULL. MaxNumTrustRegionSubProblemItr set to 10.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 10;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumTrustRegionSubProblemItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

int parseFiniteDifferenceDiagnosticsUpperSuperScripts(const mxArray* input_)
{
    int output = 0;
    if(mxGetField(input_, 0, "FiniteDifferenceDiagnosticsUpperSuperScripts") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> FiniteDifferenceDiagnosticsUpperSuperScripts keyword is NULL. FiniteDifferenceDiagnosticsUpperSuperScripts set to 5.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 5;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "FiniteDifferenceDiagnosticsUpperSuperScripts"));
        output = static_cast<int>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

int parseFiniteDifferenceDiagnosticsLowerSuperScripts(const mxArray* input_)
{
    int output = 0;
    if(mxGetField(input_, 0, "FiniteDifferenceDiagnosticsLowerSuperScripts") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> FiniteDifferenceDiagnosticsLowerSuperScripts keyword is NULL. FiniteDifferenceDiagnosticsLowerSuperScripts set to -5.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = -5;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "FiniteDifferenceDiagnosticsLowerSuperScripts"));
        output = static_cast<int>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

double parseGradientTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "GradientTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> GradientTolerance keyword is NULL. GradientTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-8;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "GradientTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseResidualTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "ResidualTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ResidualTolerance keyword is NULL. ResidualTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-8;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ResidualTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseObjectiveTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "ObjectiveTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ObjectiveTolerance keyword is NULL. ObjectiveTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-8;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ObjectiveTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseStepTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "StepTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> StepTolerance keyword is NULL. StepTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-8;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "StepTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseFeasibilityTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "FeasibilityTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> FeasibilityTolerance keyword is NULL. FeasibilityTolerance set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-4;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "FeasibilityTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseActualReductionTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "ActualReductionTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ActualReductionTolerance keyword is NULL. ActualReductionTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-8;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualReductionTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseControlStagnationTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "ControlStagnationTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ControlStagnationTolerance keyword is NULL. ControlStagnationTolerance set to 1e-2.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-2;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ControlStagnationTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMaxTrustRegionRadius(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "MaxTrustRegionRadius") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxTrustRegionRadius keyword is NULL. MaxTrustRegionRadius set to 1e4.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e4;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxTrustRegionRadius"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMinTrustRegionRadius(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "MinTrustRegionRadius") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MinTrustRegionRadius keyword is NULL. MinTrustRegionRadius set to 1e-6.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-6;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MinTrustRegionRadius"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseInitialTrustRegionRadius(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "InitialTrustRegionRadius") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> InitialTrustRegionRadius keyword is NULL. InitialTrustRegionRadius set to 1e3.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e3;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "InitialTrustRegionRadius"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseTrustRegionExpansionFactor(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "TrustRegionExpansionFactor") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TrustRegionExpansionFactor keyword is NULL. TrustRegionExpansionFactor set to 2.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 2;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionExpansionFactor"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseLineSearchContractionFactor(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "LineSearchContractionFactor") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> LineSearchContractionFactor keyword is NULL. LineSearchContractionFactor set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.5;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "LineSearchContractionFactor"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseTrustRegionContractionFactor(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "TrustRegionContractionFactor") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TrustRegionContractionFactor keyword is NULL. TrustRegionContractionFactor set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.5;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionContractionFactor"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseLineSearchStagnationTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "LineSearchStagnationTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> LineSearchStagnationTolerance keyword is NULL. LineSearchStagnationTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-8;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "LineSearchStagnationTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseFeasibleStepContractionFactor(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "FeasibleStepContractionFactor") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> FeasibleStepContractionFactor keyword is NULL. FeasibleStepContractionFactor set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.5;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "FeasibleStepContractionFactor"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMinActualOverPredictedReductionRatio(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "MinActualOverPredictedReductionRatio") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MinActualOverPredictedReductionRatio keyword is NULL. MinActualOverPredictedReductionRatio set to 0.1.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.1;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MinActualOverPredictedReductionRatio"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMidActualOverPredictedReductionRatio(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "MidActualOverPredictedReductionRatio") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MidActualOverPredictedReductionRatio keyword is NULL. MidActualOverPredictedReductionRatio set to 0.25.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.25;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MidActualOverPredictedReductionRatio"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMaxActualOverPredictedReductionRatio(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "MaxActualOverPredictedReductionRatio") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxActualOverPredictedReductionRatio keyword is NULL. MaxActualOverPredictedReductionRatio set to 0.75.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.75;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxActualOverPredictedReductionRatio"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

bool parseSetInitialTrustRegionRadiusToNormGradFlag(const mxArray* input_)
{
    bool output = false;
    if(mxGetField(input_, 0, "SetInitialTrustRegionRadiusToNormGrad") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> SetInitialTrustRegionRadiusToNormGrad keyword is NULL. SetInitialTrustRegionRadiusToNormGrad set to true.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = true;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "SetInitialTrustRegionRadiusToNormGrad"));
        std::string flag(mxArrayToString(value));
        if(flag.compare("false") == 0)
        {
            output = false;
        }
        else if(flag.compare("true") == 0)
        {
            output = true;
        }
        else
        {
            std::ostringstream msg;
            msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                    << ", -> SetInitialTrustRegionRadiusToNormGrad keyword is NOT DEFINED. Options are true or false. Default = true.\n";
            mexWarnMsgTxt(msg.str().c_str());
            output = true;
        }
    }
    return (output);
}

dotk::types::problem_t parseProblemType(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ProblemType") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ProblemType keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ProblemType"));
    dotk::types::problem_t output = dotk::mex::getProblemType(value);
    mxDestroyArray(value);
    return (output);
}

dotk::types::line_search_t parseLineSearchMethod(const mxArray* input_)
{
    dotk::types::line_search_t output = dotk::types::BACKTRACKING_CUBIC_INTRP;
    if(mxGetField(input_, 0, "LineSearchMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> LineSearchMethod keyword is NULL. LineSearchMethod set to BACKTRACKING CUBIC INTRP.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "LineSearchMethod"));
        output = dotk::mex::getLineSearchMethod(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::trustregion_t parseTrustRegionMethod(const mxArray* input_)
{
    dotk::types::trustregion_t output = dotk::types::TRUST_REGION_DOGLEG;
    if(mxGetField(input_, 0, "TrustRegionMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TrustRegionMethod keyword is NULL. TrustRegionMethod set to DOGLEG.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionMethod"));
        output = dotk::mex::getTrustRegionMethod(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::nonlinearcg_t parseNonlinearCgMethod(const mxArray* input_)
{
    dotk::types::nonlinearcg_t output = dotk::types::HAGER_ZHANG_NLCG;
    if(mxGetField(input_, 0, "NonlinearCgMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> NonlinearCgMethod keyword is NULL. NonlinearCgMethod set to HAGER-ZHANG.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NonlinearCgMethod"));
        output = dotk::mex::getNonlinearCgMethod(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::hessian_t parseHessianComputationMethod(const mxArray* input_)
{
    if(mxGetField(input_, 0, "HessianComputationMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> HessianComputationMethod keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "HessianComputationMethod"));
    dotk::types::hessian_t output = dotk::mex::getHessianComputationMethod(value);
    mxDestroyArray(value);
    return (output);
}

dotk::types::gradient_t parseGradientComputationMethod(const mxArray* input_)
{
    if(mxGetField(input_, 0, "GradientComputationMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> GradientComputationMethod keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "GradientComputationMethod"));
    dotk::types::gradient_t output = dotk::mex::getGradientComputationMethod(value);
    mxDestroyArray(value);
    return (output);
}

dotk::types::constraint_method_t parseBoundConstraintMethod(const mxArray* input_)
{
    if(mxGetField(input_, 0, "BoundConstraintMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> BoundConstraintMethod keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "BoundConstraintMethod"));
    dotk::types::constraint_method_t output = dotk::mex::getBoundConstraintMethod(value);
    mxDestroyArray(value);
    return (output);
}

mxArray* parseObjectiveFunction(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ObjectiveFunction") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ObjectiveFunction keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "ObjectiveFunction"));
    return (output);
}

mxArray* parseEqualityConstraint(const mxArray* input_)
{
    if(mxGetField(input_, 0, "EqualityConstraint") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> EqualityConstraint keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "EqualityConstraint"));
    return (output);
}

mxArray* parseInequalityConstraint(const mxArray* input_)
{
    if(mxGetField(input_, 0, "InequalityConstraint") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> InequalityConstraint keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "InequalityConstraint"));
    return (output);
}

mxArray* parseDualLowerBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "DualLowerBounds") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> DualLowerBounds keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "DualLowerBounds"));
    return (output);
}

mxArray* parseDualUpperBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "DualUpperBounds") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> DualUpperBounds keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "DualUpperBounds"));
    return (output);
}

mxArray* parseStateLowerBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "StateLowerBounds") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> StateLowerBounds keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "StateLowerBounds"));
    return (output);
}

mxArray* parseStateUpperBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "StateUpperBounds") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> StateUpperBounds keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "StateUpperBounds"));
    return (output);
}

mxArray* parseInitialControl(const mxArray* input_)
{
    if(mxGetField(input_, 0, "Control") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> Control keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "Control"));
    return (output);
}

mxArray* parseControlLowerBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ControlLowerBounds") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlLowerBounds keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "ControlLowerBounds"));
    return (output);
}

mxArray* parseControlUpperBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ControlUpperBounds") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlUpperBounds keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "ControlUpperBounds"));
    return (output);
}

mxArray* parseFiniteDifferencePerturbation(const mxArray* input_)
{
    if(mxGetField(input_, 0, "FiniteDifferencePerturbations") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> FiniteDifferencePerturbations keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "FiniteDifferencePerturbations"));
    return (output);
}

}

}
