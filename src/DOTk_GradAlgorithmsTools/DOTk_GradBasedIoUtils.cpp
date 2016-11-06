/*
 * DOTk_GradBasedIoUtils.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "vector.hpp"
#include "DOTk_VariablesUtils.hpp"
#include "DOTk_OptimizationDataMng.hpp"
#include "DOTk_GradBasedIoUtils.hpp"

namespace dotk
{

namespace ioUtils
{

bool printMessage(std::ostringstream & msg_)
{
    bool message_printed = false;
    if(msg_.str().empty() == false)
    {
        message_printed = true;
        std::cout << msg_.str().c_str() << std::flush;
    }
    return (message_printed);
}

void getLicenseMessage(std::ostringstream & msg_)
{
    std::string msg("\n\nCopyright 2014 Miguel A. Aguilo Valentin.\n\n");
    msg_ << msg.c_str();
}

void checkType(dotk::types::variable_t input_type_, dotk::types::variable_t primal_type_)
{
    try
    {
        if(input_type_ != primal_type_)
        {
            std::ostringstream msg;
            msg << "DOTk ERROR: DOTK VARIABLE TYPE MISMATCH. INPUT TYPE IS EQUAL TO " << input_type_
                    << " AND PRIMAL TYPE IS EQUAL TO " << primal_type_ << ": ABORT\n\n";
            throw msg.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

void checkDataPtr(const std::tr1::shared_ptr<dotk::vector<Real> > & data_, std::ostringstream & data_type_)
{
    try
    {
        if(data_.use_count() == 0)
        {
            std::ostringstream msg;
            msg << "DOTK ERROR: " << data_type_.str().c_str() << " DATA HAS NOT BEEN ALLOCATED: ABORT\n\n";
            throw msg.str().c_str();
        }
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

void getSolverExitCriterion(dotk::types::solver_stop_criterion_t type_, std::ostringstream & criterion_)
{
    switch(type_)
    {
        case dotk::types::NaN_CURVATURE_DETECTED:
        {
            criterion_ << "NaNCurv";
            break;
        }
        case dotk::types::ZERO_CURVATURE_DETECTED:
        {
            criterion_ << "ZeroCurv";
            break;
        }
        case dotk::types::NEGATIVE_CURVATURE_DETECTED:
        {
            criterion_ << "NegCurv";
            break;
        }
        case dotk::types::INF_CURVATURE_DETECTED:
        {
            criterion_ << "InfCurv";
            break;
        }
        case dotk::types::SOLVER_TOLERANCE_SATISFIED:
        {
            criterion_ << "Tolerance";
            break;
        }
        case dotk::types::TRUST_REGION_VIOLATED:
        {
            criterion_ << "TrustReg";
            break;
        }
        case dotk::types::MAX_SOLVER_ITR_REACHED:
        {
            criterion_ << "MaxItr";
            break;
        }
        case dotk::types::SOLVER_DID_NOT_CONVERGED:
        {
            criterion_ << "NotCnvg";
            break;
        }
        case dotk::types::NaN_RESIDUAL_NORM:
        {
            criterion_ << "NaNResNorm";
            break;
        }
        case dotk::types::INF_RESIDUAL_NORM:
        {
            criterion_ << "InfResNorm";
            break;
        }
        case dotk::types::INVALID_INEXACTNESS_MEASURE:
        {
            criterion_ << "InvalInxMeas";
            break;
        }
        case dotk::types::INVALID_ORTHOGONALITY_MEASURE:
        {
            criterion_ << "InvlOrthoMeas";
            break;
        }
    }
}

}

}
