/*
 * DOTk_ClassicalGramSchmidt.cpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "matrix.hpp"
#include "DOTk_QR.hpp"
#include "DOTk_ClassicalGramSchmidt.hpp"

namespace dotk
{

DOTk_ClassicalGramSchmidt::DOTk_ClassicalGramSchmidt() :
        dotk::DOTk_OrthogonalFactorization(dotk::types::CLASSICAL_GRAM_SCHMIDT_QR)
{
}

DOTk_ClassicalGramSchmidt::~DOTk_ClassicalGramSchmidt()
{
}

void DOTk_ClassicalGramSchmidt::factorization(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                                              std::tr1::shared_ptr<dotk::matrix<Real> > & Q_,
                                              std::tr1::shared_ptr<dotk::matrix<Real> > & R_)
{
    dotk::qr::classicalGramSchmidt(*input_, *Q_, *R_);
}

}
