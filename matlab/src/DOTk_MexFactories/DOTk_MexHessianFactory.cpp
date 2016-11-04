/*
 * DOTk_MexHessianFactory.cpp
 *
 *  Created on: Oct 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_MexHessianFactory.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexQuasiNewtonParser.hpp"

namespace dotk
{

namespace mex
{

void buildHessian(const mxArray* options_,
                  const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                  std::tr1::shared_ptr<dotk::DOTk_Hessian> & hessian_)
{
    size_t storage = 0;
    dotk::types::hessian_t type = dotk::types::HESSIAN_DISABLED;
    dotk::mex::parseHessianComputationMethod(options_, type);
    switch(type)
    {
        case dotk::types::LBFGS_HESS:
        {
            dotk::mex::parseQuasiNewtonStorage(options_, storage);
            hessian_->setLbfgsHessian(primal_->control(), storage);
            break;
        }
        case dotk::types::LDFP_HESS:
        {
            dotk::mex::parseQuasiNewtonStorage(options_, storage);
            hessian_->setLdfpHessian(primal_->control(), storage);
            break;
        }
        case dotk::types::LSR1_HESS:
        {
            dotk::mex::parseQuasiNewtonStorage(options_, storage);
            hessian_->setLsr1Hessian(primal_->control(), storage);
            break;
        }
        case dotk::types::SR1_HESS:
        {
            hessian_->setSr1Hessian(primal_->control());
            break;
        }
        case dotk::types::DFP_HESS:
        {
            hessian_->setDfpHessian(primal_->control());
            break;
        }
        case dotk::types::USER_DEFINED_HESS:
        {
            hessian_->setReducedSpaceHessian();
            break;
        }
        case dotk::types::BARZILAIBORWEIN_HESS:
        {
            hessian_->setBarzilaiBorweinHessian(primal_->control());
            break;
        }
        case dotk::types::USER_DEFINED_HESS_TYPE_CNP:
        {
            hessian_->setFullSpaceHessian();
            break;
        }
        case dotk::types::HESSIAN_DISABLED:
        default:
        {
            std::string msg(" DOTk/MEX ERROR: Invalid Hessian Method. See Users' Manual. \n");
            mexErrMsgTxt(msg.c_str());
            break;
        }
    }
}

}

}
