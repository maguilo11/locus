/*
 * DOTk_MexKrylovSolverParser.cpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include <string>
#include <sstream>
#include "DOTk_MexKrylovSolverParser.hpp"

namespace dotk
{

namespace mex
{

dotk::types::krylov_solver_t getKrylovSolverMethod(const mxArray* input_)
{
    std::string method(mxArrayToString(input_));
    dotk::types::krylov_solver_t type = dotk::types::KRYLOV_SOLVER_DISABLED;

    if(method.compare("PCG") == 0)
    {
        mexPrintf(" KrylovSolverMethod = CG \n");
        type = dotk::types::LEFT_PREC_CG;
    }
    else if(method.compare("GMRES") == 0)
    {
        mexPrintf(" KrylovSolverMethod = GMRES \n");
        type = dotk::types::PREC_GMRES;
    }
    else if(method.compare("PCGNR") == 0)
    {
        mexPrintf(" KrylovSolverMethod = CGNR \n");
        type = dotk::types::LEFT_PREC_CGNR;
    }
    else if(method.compare("PCGNE") == 0)
    {
        mexPrintf(" KrylovSolverMethod = CGNE \n");
        type = dotk::types::LEFT_PREC_CGNE;
    }
    else if(method.compare("PCR") == 0)
    {
        mexPrintf(" KrylovSolverMethod = CR \n");
        type = dotk::types::LEFT_PREC_CR;
    }
    else if(method.compare("PGCR") == 0)
    {
        mexPrintf(" KrylovSolverMethod = GCR \n");
        type = dotk::types::LEFT_PREC_GCR;
    }
    else if(method.compare("PROJECTED_PCG") == 0)
    {
        mexPrintf(" KrylovSolverMethod = PROJECTED CG \n");
        type = dotk::types::PROJECTED_PREC_CG;
    }
    else if(method.compare("USER_DEFINED") == 0)
    {
        mexPrintf(" KrylovSolverMethod = USER DEFINED \n");
        type = dotk::types::USER_DEFINED_KRYLOV_SOLVER;
    }
    else
    {
        type = dotk::types::LEFT_PREC_CG;
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> KrylovSolverMethod keyword is misspelled. KrylovSolverMethod set to PCG.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }

    return (type);
}

size_t parseMaxNumKrylovSolverItr(const mxArray* input_)
{
    size_t output = 200;
    if(mxGetField(input_, 0, "MaxNumKrylovSolverItr") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> MaxNumKrylovSolverItr keyword is NULL. MaxNumKrylovSolverItr set to 10.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "MaxNumKrylovSolverItr"));
        output = static_cast<size_t>(mxGetScalar(value));
        mxDestroyArray(value);
    }
    return (output);
}

double parseKrylovSolverFixTolerance(const mxArray* input_)
{
    double output = 1e-8;
    if(mxGetField(input_, 0, "FixTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> FixTolerance keyword is NULL. FixTolerance set to 1e-8.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "FixTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseRelativeToleranceExponential(const mxArray* input_)
{
    double output = 0.5;
    if(mxGetField(input_, 0, "RelativeToleranceExponential") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> RelativeToleranceExponential keyword is NULL. RelativeToleranceExponential set to 0.5.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "RelativeToleranceExponential"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

double parseKrylovSolverRelativeTolerance(const mxArray* input_)
{
    double output = 1e-2;
    if(mxGetField(input_, 0, "RelativeTolerance") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> RelativeTolerance keyword is NULL. RelativeTolerance set to 1e-2.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "RelativeTolerance"));
        output = mxGetScalar(value);
        mxDestroyArray(value);
    }
    return (output);
}

dotk::types::krylov_solver_t parseKrylovSolverMethod(const mxArray* input_)
{
    dotk::types::krylov_solver_t output = dotk::types::LEFT_PREC_CG;
    if(mxGetField(input_, 0, "KrylovSolverMethod") == nullptr)
    {
        std::ostringstream msg;
        msg << "\nWARNING IN: " << __FILE__ << ", LINE: " << __LINE__
                << ", -> KrylovSolverMethod keyword is NULL. KrylovSolverMethod set to PCG.\n";
        mexWarnMsgTxt(msg.str().c_str());
    }
    else
    {
        mxArray* value = mxDuplicateArray(mxGetField(input_, 0, "KrylovSolverMethod"));
        output = dotk::mex::getKrylovSolverMethod(value);
        mxDestroyArray(value);
    }
    return (output);
}

}

}
