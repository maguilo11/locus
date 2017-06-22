/*
 * DOTk_MexHessianFactory.cpp
 *
 *  Created on: Oct 15, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "DOTk_Primal.hpp"
#include "DOTk_Hessian.hpp"
#include "DOTk_MexVector.hpp"
#include "DOTk_MexHessianFactory.hpp"
#include "DOTk_MexAlgorithmParser.hpp"
#include "DOTk_MexQuasiNewtonParser.hpp"

namespace dotk
{

namespace mex
{

void buildHessian(const mxArray* options_,
                  std::shared_ptr<dotk::DOTk_Hessian> & hessian_)
{
    dotk::types::hessian_t type = dotk::mex::parseHessianComputationMethod(options_);
    switch(type)
    {
        case dotk::types::LBFGS_HESS:
        {
            size_t num_controls = dotk::mex::parseNumberControls(options_);
            dotk::MexVector work(num_controls, 0.);
            size_t storage = dotk::mex::parseQuasiNewtonStorage(options_);
            hessian_->setLbfgsHessian(work, storage);
            break;
        }
        case dotk::types::LDFP_HESS:
        {
            size_t num_controls = dotk::mex::parseNumberControls(options_);
            dotk::MexVector work(num_controls, 0.);
            size_t storage = dotk::mex::parseQuasiNewtonStorage(options_);
            hessian_->setLdfpHessian(work, storage);
            break;
        }
        case dotk::types::LSR1_HESS:
        {
            size_t num_controls = dotk::mex::parseNumberControls(options_);
            dotk::MexVector work(num_controls, 0.);
            size_t storage = dotk::mex::parseQuasiNewtonStorage(options_);
            hessian_->setLsr1Hessian(work, storage);
            break;
        }
        case dotk::types::SR1_HESS:
        {
            size_t num_controls = dotk::mex::parseNumberControls(options_);
            dotk::MexVector work(num_controls, 0.);
            hessian_->setSr1Hessian(work);
            break;
        }
        case dotk::types::DFP_HESS:
        {
            size_t num_controls = dotk::mex::parseNumberControls(options_);
            dotk::MexVector work(num_controls, 0.);
            hessian_->setDfpHessian(work);
            break;
        }
        case dotk::types::USER_DEFINED_HESS:
        {
            hessian_->setReducedSpaceHessian();
            break;
        }
        case dotk::types::BARZILAIBORWEIN_HESS:
        {
            size_t num_controls = dotk::mex::parseNumberControls(options_);
            dotk::MexVector work(num_controls, 0.);
            hessian_->setBarzilaiBorweinHessian(work);
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
            std::ostringstream msg;
            msg << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", -> UNDEFINED Hessian Method.\n";
            mexErrMsgTxt(msg.str().c_str());
            break;
        }
    }
}

}

}
