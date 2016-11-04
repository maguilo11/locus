/*
 * DOTk_MexOptimalityCriteriaParser.cpp
 *
 *  Created on: Jun 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include "DOTk_MexOptimalityCriteriaParser.hpp"

namespace dotk
{

namespace mex
{

void parseOptCriteriaMoveLimit(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MoveLimit")) == true)
    {
        output_ = 0.1;
        std::string msg(" DOTk/MEX ERROR: MoveLimit is NOT Defined. Default = 0.01. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "MoveLimit")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseOptCriteriaDualLowerBound(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualLowerBound")) == true)
    {
        output_ = 0.1;
        std::string msg(" DOTk/MEX ERROR: DualLowerBound is NOT Defined. Default = 0. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "DualLowerBound")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseOptCriteriaDualUpperBound(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DualUpperBound")) == true)
    {
        output_ = 0.1;
        std::string msg(" DOTk/MEX ERROR: DualUpperBound is NOT Defined. Default = 1e4. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "DualUpperBound")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseOptCriteriaDampingParameter(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "DampingParameter")) == true)
    {
        output_ = 0.1;
        std::string msg(" DOTk/MEX ERROR: DampingParameter is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "DampingParameter")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

void parseOptCriteriaBisectionTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "BisectionTolerance")) == true)
    {
        output_ = 0.1;
        std::string msg(" DOTk/MEX ERROR: BisectionTolerance is NOT Defined. Default = 1e-4. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr factor;
    factor.reset(mxDuplicateArray(mxGetField(options_, 0, "BisectionTolerance")));
    output_ = mxGetScalar(factor.get());
    factor.release();
}

}

}
