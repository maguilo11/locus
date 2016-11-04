/*
 * DOTk_MexKrylovSolverParser.cpp
 *
 *  Created on: Apr 19, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include "DOTk_MexArrayPtr.hpp"
#include "DOTk_MexKrylovSolverParser.hpp"

namespace dotk
{

namespace mex
{

dotk::types::krylov_solver_t getKrylovSolverMethod(dotk::DOTk_MexArrayPtr & ptr_)
{
    std::string method(mxArrayToString(ptr_.get()));
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
        std::string msg(" DOTk/MEX WARNING: Invalid Krylov Solver Method. Default = CONJUGATE GRADIENT. \n");
        mexWarnMsgTxt(msg.c_str());
    }

    return (type);
}

void parseMaxNumKrylovSolverItr(const mxArray* options_, size_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "MaxNumKrylovSolverItr")) == true)
    {
        output_ = 200;
        std::string msg(" DOTk/MEX WARNING: MaxNumKrylovSolverItr is NOT Defined. Default = 200. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr iterations;
    iterations.reset(mxDuplicateArray(mxGetField(options_, 0, "MaxNumKrylovSolverItr")));
    output_ = static_cast<size_t>(mxGetScalar(iterations.get()));
    iterations.release();
}

void parseKrylovSolverFixTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "FixTolerance")) == true)
    {
        output_ = 1e-8;
        std::string msg(" DOTk/MEX WARNING: FixTolerance is NOT Defined. Default = 1e-8. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "FixTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseRelativeToleranceExponential(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "RelativeToleranceExponential")) == true)
    {
        output_ = 0.5;
        std::string msg(" DOTk/MEX WARNING: RelativeToleranceExponential is NOT Defined. Default = 0.5. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr exponential;
    exponential.reset(mxDuplicateArray(mxGetField(options_, 0, "RelativeToleranceExponential")));
    output_ = mxGetScalar(exponential.get());
    exponential.release();
}

void parseKrylovSolverRelativeTolerance(const mxArray* options_, double & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "RelativeTolerance")) == true)
    {
        output_ = 1e-2;
        std::string msg(" DOTk/MEX WARNING: RelativeTolerance is NOT Defined. Default = 1e-2. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr tolerance;
    tolerance.reset(mxDuplicateArray(mxGetField(options_, 0, "RelativeTolerance")));
    output_ = mxGetScalar(tolerance.get());
    tolerance.release();
}

void parseKrylovSolverMethod(const mxArray* options_, dotk::types::krylov_solver_t & output_)
{
    if(mxIsEmpty(mxGetField(options_, 0, "KrylovSolverMethod")) == true)
    {
        output_ = dotk::types::LEFT_PREC_CG;
        std::string msg(" DOTk/MEX WARNING: KrylovSolverMethod is NOT Defined. Default = CONJUGATE GRADIENT. \n");
        mexWarnMsgTxt(msg.c_str());
        return;
    }
    dotk::DOTk_MexArrayPtr type;
    type.reset(mxDuplicateArray(mxGetField(options_, 0, "KrylovSolverMethod")));
    output_ = dotk::mex::getKrylovSolverMethod(type);
    type.release();
}

}

}
