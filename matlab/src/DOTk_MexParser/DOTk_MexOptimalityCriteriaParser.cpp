/*
 * DOTk_MexOptimalityCriteriaParser.cpp
 *
 *  Created on: Jun 27, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>

#include "DOTk_MexOptimalityCriteriaParser.hpp"

namespace dotk
{

namespace mex
{

double parseOptCriteriaMoveLimit(const mxArray* input_)
{
    double output = 0.01;
    if(mxGetField(input_, 0, "MoveLimit") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MoveLimit keyword is NULL. MoveLimit set to 0.01\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MoveLimit"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseOptCriteriaDualLowerBound(const mxArray* input_)
{
    double output = 0.;
    if(mxGetField(input_, 0, "DualLowerBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualLowerBound keyword is NULL. DualLowerBound set to 0.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualLowerBound"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseOptCriteriaDualUpperBound(const mxArray* input_)
{
    double output = 1e4;
    if(mxGetField(input_, 0, "DualUpperBound") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DualUpperBound keyword is NULL. DualUpperBound set to 1e4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DualUpperBound"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseOptCriteriaDampingParameter(const mxArray* input_)
{
    double output = 0.5;
    if(mxGetField(input_, 0, "DampingParameter") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> DampingParameter keyword is NULL. DampingParameter set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "DampingParameter"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseOptCriteriaBisectionTolerance(const mxArray* input_)
{
    double output = 1e-4;
    if(mxGetField(input_, 0, "BisectionTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> BisectionTolerance keyword is NULL. BisectionTolerance set to 1e-4.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "BisectionTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

}

}
