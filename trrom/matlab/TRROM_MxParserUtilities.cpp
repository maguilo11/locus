/*
 * TRROM_MxParserUtilities.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <sstream>

#include "TRROM_MxParserUtilities.hpp"

namespace trrom
{

namespace mx
{

int parseNumberControls(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberControls") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberControls keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberControls"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseNumberStates(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberStates") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberStates keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberStates"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseNumberDuals(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberDuals") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberDuals keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberDuals"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseNumberSlacks(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberSlacks") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> NumberSlacks keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "NumberSlacks"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseMaxNumberSubProblemIterations(const mxArray* input_)
{
    int output = 0;
    if(mxGetField(input_, 0, "MaxNumberSubProblemIterations") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumberSubProblemIterations keyword is NULL. MaxNumberSubProblemIterations set to 10.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 10;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberSubProblemIterations"));
        output = static_cast<int>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

int parseMaxNumberOuterIterations(const mxArray* input_)
{
    int output = 0;
    if(mxGetField(input_, 0, "MaxNumberOuterIterations") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumberSubProblemIterations keyword is NULL. MaxNumberOuterIterations set to 50.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 50;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberOuterIterations"));
        output = static_cast<int>(mxGetScalar(value));
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
                << ", -> MinTrustRegionRadius keyword is NULL. MinTrustRegionRadius set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-4;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MinTrustRegionRadius"));
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

double parseTrustRegionContractionScalar(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "TrustRegionContractionScalar") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TrustRegionContractionScalar keyword is NULL. TrustRegionContractionScalar set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.5;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionContractionScalar"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseTrustRegionExpansionScalar(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "TrustRegionExpansionScalar") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TrustRegionExpansionScalar keyword is NULL. TrustRegionExpansionScalar set to 2.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 2;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionExpansionScalar"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseActualOverPredictedReductionMidBound(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "ActualOverPredictedReductionMidBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ActualOverPredictedReductionMidBound keyword is NULL. ActualOverPredictedReductionMidBound set to 0.25.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.25;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionMidBound"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseActualOverPredictedReductionLowerBound(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "ActualOverPredictedReductionLowerBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ActualOverPredictedReductionLowerBound keyword is NULL. ActualOverPredictedReductionLowerBound set to 0.1.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.1;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionLowerBound"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseActualOverPredictedReductionUpperBound(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "ActualOverPredictedReductionUpperBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ActualOverPredictedReductionUpperBound keyword is NULL. ActualOverPredictedReductionUpperBound set to 0.75.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 0.75;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionUpperBound"));
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
                << ", -> StepTolerance keyword is NULL. StepTolerance set to 1e-10.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-10;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "StepTolerance"));
        output = mxGetScalar(value);
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
                << ", -> GradientTolerance keyword is NULL. GradientTolerance set to 1e-10.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-10;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "GradientTolerance"));
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
                << ", -> ObjectiveTolerance keyword is NULL. ObjectiveTolerance set to 1e-10.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-10;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ObjectiveTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseStagnationTolerance(const mxArray* input_)
{
    double output = 0;
    if(mxGetField(input_, 0, "StagnationTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> StagnationTolerance keyword is NULL. StagnationTolerance set to 1e-12.\n";
        mexWarnMsgTxt(msg.str().c_str());
        output = 1e-12;
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "StagnationTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

mxArray* parseControlLowerBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ControlLowerBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlLowerBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "ControlLowerBound"));
    return (output);
}

mxArray* parseControlUpperBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ControlUpperBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlUpperBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "ControlUpperBound"));
    return (output);
}

mxArray* parseReducedBasisObjectiveFunction(const mxArray* input_)
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

mxArray* parseReducedBasisPartialDifferentialEquation(const mxArray* input_)
{
    if(mxGetField(input_, 0, "PartialDifferentialEquation") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> PartialDifferentialEquation keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* output = mxDuplicateArray(mxGetField(input_, 0, "PartialDifferentialEquation"));
    return (output);
}

}

}
