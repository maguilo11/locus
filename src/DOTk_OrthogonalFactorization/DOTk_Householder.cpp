/*
 * DOTk_Householder.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "matrix.hpp"
#include "DOTk_QR.hpp"
#include "DOTk_Householder.hpp"

namespace dotk
{

DOTk_Householder::DOTk_Householder() :
        dotk::DOTk_OrthogonalFactorization(dotk::types::HOUSEHOLDER_QR)
{
}

DOTk_Householder::~DOTk_Householder()
{
}

void DOTk_Householder::factorization(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                                     std::tr1::shared_ptr<dotk::matrix<Real> > & Q_,
                                     std::tr1::shared_ptr<dotk::matrix<Real> > & R_)
{
    dotk::qr::householder(*input_, *Q_, *R_);
}

}
