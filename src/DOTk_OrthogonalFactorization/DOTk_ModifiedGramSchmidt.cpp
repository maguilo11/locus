/*
 * DOTk_ModifiedGramSchmidt.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "matrix.hpp"
#include "DOTk_QR.hpp"
#include "DOTk_ModifiedGramSchmidt.hpp"

namespace dotk
{

DOTk_ModifiedGramSchmidt::DOTk_ModifiedGramSchmidt() :
        dotk::DOTk_OrthogonalFactorization(dotk::types::MODIFIED_GRAM_SCHMIDT_QR)
{
}

DOTk_ModifiedGramSchmidt::~DOTk_ModifiedGramSchmidt()
{
}

void DOTk_ModifiedGramSchmidt::factorization(const std::shared_ptr<dotk::matrix<Real> > & input_,
                                             std::shared_ptr<dotk::matrix<Real> > & Q_,
                                             std::shared_ptr<dotk::matrix<Real> > & R_)
{
    dotk::qr::modifiedGramSchmidt(*input_, *Q_, *R_);
}

}
