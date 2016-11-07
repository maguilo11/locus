/*
 * DOTk_MexParseAlgorithmTypeSQP.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>

#include "DOTk_MexParseAlgorithmTypeSQP.hpp"

namespace dotk
{

namespace mex
{

void parseSqpMaxNumDualProblemItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumDualProblemItr")) == true)
    {
        output_ = 200;
        std::string msg(" DOTk/MEX ERROR: MaxNumDualProblemItr is NOT Defined. Default = 200. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr itr;
    itr.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumDualProblemItr")));
    output_ = static_cast<size_t>(mxGetScalar(itr.get()));
    itr.release();
}

void parseSqpMaxNumTangentialProblemItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumTangentialProblemItr")) == true)
    {
        output_ = 200;
        std::string msg(" DOTk/MEX ERROR: MaxNumTangentialProblemItr is NOT Defined. Default = 200. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr itr;
    itr.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumTangentialProblemItr")));
    output_ = static_cast<size_t>(mxGetScalar(itr.get()));
    itr.release();
}

void parseSqpMaxNumQuasiNormalProblemItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumQuasiNormalProblemItr")) == true)
    {
        output_ = 200;
        std::string msg(" DOTk/MEX ERROR: MaxNumQuasiNormalProblemItr is NOT Defined. Default = 200. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr itr;
    itr.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumQuasiNormalProblemItr")));
    output_ = static_cast<size_t>(mxGetScalar(itr.get()));
    itr.release();
}

void parseSqpMaxNumTangentialSubProblemItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumTangentialSubProblemItr")) == true)
    {
        output_ = 200;
        std::string msg(" DOTk/MEX ERROR: MaxNumTangentialSubProblemItr is NOT Defined. Default = 200. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr itr;
    itr.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumTangentialSubProblemItr")));
    output_ = static_cast<size_t>(mxGetScalar(itr.get()));
    itr.release();
}

void parseTangentialTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "TangentialTolerance")) == true)
    {
        output_ = 1e-4;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "TangentialTolerance")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseDualProblemTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualProblemTolerance")) == true)
    {
        output_ = 1e-4;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "DualProblemTolerance")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseDualDotGradientTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualDotGradientTolerance")) == true)
    {
        output_ = 1e4;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "DualDotGradientTolerance")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseToleranceContractionFactor(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ToleranceContractionFactor")) == true)
    {
        output_ = 1e-1;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "ToleranceContractionFactor")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parsePredictedReductionParameter(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "PredictedReductionParameter")) == true)
    {
        output_ = 1e-8;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "PredictedReductionParameter")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseMeritFunctionPenaltyParameter(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MeritFunctionPenaltyParameter")) == true)
    {
        output_ = 1.;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "MeritFunctionPenaltyParameter")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseQuasiNormalProblemRelativeTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "QuasiNormalProblemRelativeTolerance")) == true)
    {
        output_ = 1e-4;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "QuasiNormalProblemRelativeTolerance")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseTangentialToleranceContractionFactor(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "TangentialToleranceContractionFactor")) == true)
    {
        output_ = 1e-3;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "TangentialToleranceContractionFactor")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseActualOverPredictedReductionTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "ActualOverPredictedReductionTolerance")) == true)
    {
        output_ = 1e-8;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "ActualOverPredictedReductionTolerance")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseMaxEffectiveTangentialOverTrialStepRatio(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxEffectiveTangentialOverTrialStepRatio")) == true)
    {
        output_ = 2.;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxEffectiveTangentialOverTrialStepRatio")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseTangentialSubProbLeftPrecProjectionTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "TangentialSubProbLeftPrecProjectionTolerance")) == true)
    {
        output_ = 1e-4;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "TangentialSubProbLeftPrecProjectionTolerance")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

void parseQuasiNormalProblemTrustRegionRadiusPenaltyParameter(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "QuasiNormalProblemTrustRegionRadiusPenaltyParameter")) == true)
    {
        output_ = 0.8;
        return;
    }
    dotk::DOTk_MexArrayPtr ptr;
    ptr.reset(mxDuplicateArray(mxGetField(options_, 0, "QuasiNormalProblemTrustRegionRadiusPenaltyParameter")));
    output_ = mxGetScalar(ptr.get());
    ptr.release();
}

}

}
