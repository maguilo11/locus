/*
 * TRROM_MxParsingUtilities.cpp
 *
 *  Created on: Dec 23, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "TRROM_MxVector.hpp"
#include "TRROM_MxReducedBasisPDE.hpp"
#include "TRROM_MxParsingUtilities.hpp"
#include "TRROM_MxReducedBasisObjective.hpp"

namespace trrom
{

namespace mx
{

int parseNumberControls(const mxArray* input_)
{
    if(mxGetField(input_, 0, "NumberControls") == NULL)
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
    if(mxGetField(input_, 0, "NumberStates") == NULL)
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
    if(mxGetField(input_, 0, "NumberDuals") == NULL)
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
    if(mxGetField(input_, 0, "NumberSlacks") == NULL)
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
    if(mxGetField(input_, 0, "MaxNumberSubProblemIterations") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MaxNumberSubProblemIterations keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberSubProblemIterations"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

int parseMaxNumberOuterIterations(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MaxNumberOuterIterations") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MaxNumberOuterIterations keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumberOuterIterations"));
    int output = static_cast<int>(mxGetScalar(value));
    mxDestroyArray(value);

    return (output);
}

double parseMinTrustRegionRadius(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MinTrustRegionRadius") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MinTrustRegionRadius keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MinTrustRegionRadius"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseMaxTrustRegionRadius(const mxArray* input_)
{
    if(mxGetField(input_, 0, "MaxTrustRegionRadius") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> MaxTrustRegionRadius keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxTrustRegionRadius"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseTrustRegionContractionScalar(const mxArray* input_)
{
    if(mxGetField(input_, 0, "TrustRegionContractionScalar") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> TrustRegionContractionScalar keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionContractionScalar"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseTrustRegionExpansionScalar(const mxArray* input_)
{
    if(mxGetField(input_, 0, "TrustRegionExpansionScalar") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> TrustRegionExpansionScalar keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TrustRegionExpansionScalar"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseActualOverPredictedReductionMidBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ActualOverPredictedReductionMidBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ActualOverPredictedReductionMidBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionMidBound"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseActualOverPredictedReductionLowerBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ActualOverPredictedReductionLowerBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ActualOverPredictedReductionLowerBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionLowerBound"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseActualOverPredictedReductionUpperBound(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ActualOverPredictedReductionUpperBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ActualOverPredictedReductionUpperBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionUpperBound"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseStepTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "StepTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> StepTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "StepTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseGradientTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "GradientTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> GradientTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "GradientTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseObjectiveTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "ObjectiveTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ObjectiveTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ObjectiveTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

double parseStagnationTolerance(const mxArray* input_)
{
    if(mxGetField(input_, 0, "StagnationTolerance") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> StagnationTolerance keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "StagnationTolerance"));
    int output = mxGetScalar(value);
    mxDestroyArray(value);

    return (output);
}

void parseControlLowerBound(const mxArray* input_, trrom::MxVector & output_)
{
    if(mxGetField(input_, 0, "ControlLowerBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlLowerBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ControlLowerBound"));
    output_.setMxArray(value);
    mxDestroyArray(value);
}

void parseControlUpperBound(const mxArray* input_, trrom::MxVector & output_)
{
    if(mxGetField(input_, 0, "ControlUpperBound") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ControlUpperBound keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ControlUpperBound"));
    output_.setMxArray(value);
    mxDestroyArray(value);
}

void parseReducedBasisObjectiveFunction(const mxArray* input_, std::tr1::shared_ptr<trrom::ReducedBasisObjective> & output_)
{
    if(mxGetField(input_, 0, "ObjectiveFunction") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> ObjectiveFunction keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* mx_objective_function = mxDuplicateArray(mxGetField(input_, 0, "ObjectiveFunction"));
    output_.reset(new trrom::MxReducedBasisObjective(mx_objective_function));
    mxDestroyArray(mx_objective_function);
}

void parseReducedBasisPartialDifferentialEquation(const mxArray* input_, std::tr1::shared_ptr<trrom::ReducedBasisPDE> & output_)
{
    if(mxGetField(input_, 0, "PartialDifferentialEquation") == NULL)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> PartialDifferentialEquation keyword is NULL.\n";
        mexErrMsgTxt(msg.str().c_str());
    }
    mxArray* mx_reduced_basis_pde = mxDuplicateArray(mxGetField(input_, 0, "PartialDifferentialEquation"));
    output_.reset(new trrom::MxReducedBasisPDE(mx_reduced_basis_pde));
    mxDestroyArray(mx_reduced_basis_pde);
}

}

}
