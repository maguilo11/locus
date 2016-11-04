/*
 * DOTk_MexFactoriesAlgorithmTypeFO.cpp
 *
 *  Created on: May 1, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_MexQuasiNewtonParser.hpp"
#include "DOTk_LineSearchQuasiNewton.hpp"
#include "DOTk_MexFactoriesAlgorithmTypeFO.hpp"

namespace dotk
{

namespace mex
{

void buildQuasiNewtonMethod(const mxArray* options_, dotk::DOTk_LineSearchQuasiNewton & algorithm_)
{
    dotk::types::invhessian_t type = dotk::types::INV_HESS_DISABLED;
    dotk::mex::parseQuasiNewtonMethod(options_, type);

    switch(type)
    {
        case dotk::types::LBFGS_INV_HESS:
        {
            size_t storage = 0;
            dotk::mex::parseQuasiNewtonStorage(options_, storage);
            algorithm_.setLbfgsSecantMethod(storage);
            break;
        }
        case dotk::types::LDFP_INV_HESS:
        {
            size_t storage = 0;
            dotk::mex::parseQuasiNewtonStorage(options_, storage);
            algorithm_.setLdfpSecantMethod(storage);
            break;
        }
        case dotk::types::LSR1_INV_HESS:
        {
            size_t storage = 0;
            dotk::mex::parseQuasiNewtonStorage(options_, storage);
            algorithm_.setLsr1SecantMethod(storage);
            break;
        }
        case dotk::types::SR1_INV_HESS:
        {
            algorithm_.setSr1SecantMethod();
            break;
        }
        case dotk::types::BFGS_INV_HESS:
        {
            algorithm_.setBfgsSecantMethod();
            break;
        }
        case dotk::types::BARZILAIBORWEIN_INV_HESS:
        {
            algorithm_.setBarzilaiBorweinSecantMethod();
            break;
        }
        case dotk::types::USER_DEFINED_INV_HESS:
        case dotk::types::INV_HESS_DISABLED:
        default:
        {
            algorithm_.setBfgsSecantMethod();
            std::string msg(" DOTk/MEX WARNING: Invalid Quasi-Newton Method. Default = BFGS. \n");
            mexWarnMsgTxt(msg.c_str());
            break;
        }
    }
}

}

}
