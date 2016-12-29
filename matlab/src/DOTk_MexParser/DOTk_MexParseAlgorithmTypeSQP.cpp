/*
 * DOTk_MexParseAlgorithmTypeSQP.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <sstream>
#include "DOTk_MexParseAlgorithmTypeSQP.hpp"

namespace dotk
{

namespace mex
{

size_t parseSqpMaxNumDualProblemItr(const mxArray* input_)
{
    size_t output = 200;
    if(mxGetField(input_, 0, "MaxNumDualProblemItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumDualProblemItr keyword is NULL. MaxNumDualProblemItr set to 200.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumTrustRegionSubProblemItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseSqpMaxNumTangentialProblemItr(const mxArray* input_)
{
    size_t output = 200;
    if(mxGetField(input_, 0, "MaxNumTangentialProblemItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumTangentialProblemItr keyword is NULL. MaxNumTangentialProblemItr set to 200.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumTangentialProblemItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseSqpMaxNumQuasiNormalProblemItr(const mxArray* input_)
{
    size_t output = 200;
    if(mxGetField(input_, 0, "MaxNumQuasiNormalProblemItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumQuasiNormalProblemItr keyword is NULL. MaxNumQuasiNormalProblemItr set to 200.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumQuasiNormalProblemItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

size_t parseSqpMaxNumTangentialSubProblemItr(const mxArray* input_)
{
    size_t output = 200;
    if(mxGetField(input_, 0, "MaxNumTangentialSubProblemItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumTangentialSubProblemItr keyword is NULL. MaxNumTangentialSubProblemItr set to 200.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumTangentialSubProblemItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

double parseTangentialTolerance(const mxArray* input_)
{
    double output = 1e-4;
    if(mxGetField(input_, 0, "TangentialTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TangentialTolerance keyword is NULL. TangentialTolerance set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TangentialTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseDualProblemTolerance(const mxArray* input_)
{
    double output = 1e-4;
    if(mxGetField(input_, 0, "DualProblemTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualProblemTolerance keyword is NULL. DualProblemTolerance set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualProblemTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseDualDotGradientTolerance(const mxArray* input_)
{
    double output = 1e-4;
    if(mxGetField(input_, 0, "DualDotGradientTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualDotGradientTolerance keyword is NULL. DualDotGradientTolerance set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualDotGradientTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseToleranceContractionFactor(const mxArray* input_)
{
    double output = 1e-1;
    if(mxGetField(input_, 0, "ToleranceContractionFactor") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ToleranceContractionFactor keyword is NULL. ToleranceContractionFactor set to 1e-1.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ToleranceContractionFactor"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parsePredictedReductionParameter(const mxArray* input_)
{
    double output = 1e-8;
    if(mxGetField(input_, 0, "PredictedReductionParameter") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> PredictedReductionParameter keyword is NULL. PredictedReductionParameter set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "PredictedReductionParameter"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMeritFunctionPenaltyParameter(const mxArray* input_)
{
    double output = 1;
    if(mxGetField(input_, 0, "MeritFunctionPenaltyParameter") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MeritFunctionPenaltyParameter keyword is NULL. MeritFunctionPenaltyParameter set to 1.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MeritFunctionPenaltyParameter"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseQuasiNormalProblemRelativeTolerance(const mxArray* input_)
{
    double output = 1e-4;
    if(mxGetField(input_, 0, "QuasiNormalProblemRelativeTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> QuasiNormalProblemRelativeTolerance keyword is NULL. QuasiNormalProblemRelativeTolerance set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "QuasiNormalProblemRelativeTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseTangentialToleranceContractionFactor(const mxArray* input_)
{
    double output = 1e-3;
    if(mxGetField(input_, 0, "TangentialToleranceContractionFactor") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TangentialToleranceContractionFactor keyword is NULL. TangentialToleranceContractionFactor set to 1e-3.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TangentialToleranceContractionFactor"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseActualOverPredictedReductionTolerance(const mxArray* input_)
{
    double output = 1e-8;
    if(mxGetField(input_, 0, "ActualOverPredictedReductionTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> ActualOverPredictedReductionTolerance keyword is NULL. ActualOverPredictedReductionTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "ActualOverPredictedReductionTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseMaxEffectiveTangentialOverTrialStepRatio(const mxArray* input_)
{
    double output = 2;
    if(mxGetField(input_, 0, "MaxEffectiveTangentialOverTrialStepRatio") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxEffectiveTangentialOverTrialStepRatio keyword is NULL. MaxEffectiveTangentialOverTrialStepRatio set to 2.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxEffectiveTangentialOverTrialStepRatio"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseTangentialSubProbLeftPrecProjectionTolerance(const mxArray* input_)
{
    double output = 1e-4;
    if(mxGetField(input_, 0, "TangentialSubProbLeftPrecProjectionTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> TangentialSubProbLeftPrecProjectionTolerance keyword is NULL."
                << " TangentialSubProbLeftPrecProjectionTolerance set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "TangentialSubProbLeftPrecProjectionTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseQuasiNormalProblemTrustRegionRadiusPenaltyParameter(const mxArray* input_)
{
    double output = 0.8;
    if(mxGetField(input_, 0, "QuasiNormalProblemTrustRegionRadiusPenaltyParameter") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> QuasiNormalProblemTrustRegionRadiusPenaltyParameter keyword is NULL."
                << " QuasiNormalProblemTrustRegionRadiusPenaltyParameter set to 0.8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "QuasiNormalProblemTrustRegionRadiusPenaltyParameter"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

}

}
