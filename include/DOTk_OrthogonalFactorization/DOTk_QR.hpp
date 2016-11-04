/*
 * DOTk_QR.hpp
 *
 *  Created on: Jul 2, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_QR_HPP_
#define DOTK_QR_HPP_

#include "matrix.hpp"

namespace dotk
{

namespace qr
{

void classicalGramSchmidt(dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_);
void classicalGramSchmidt(const dotk::matrix<Real> & A_, dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_);

void modifiedGramSchmidt(dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_);
void modifiedGramSchmidt(const dotk::matrix<Real> & A_, dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_);

void arnoldiModifiedGramSchmidt(const dotk::matrix<Real> & A_,
                                dotk::matrix<Real> & Q_,
                                dotk::matrix<Real> & Hessenberg_,
                                Real tolerance_ = std::numeric_limits<Real>::epsilon());

void householder(dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_);
void householder(const dotk::matrix<Real> & A_, dotk::matrix<Real> & Q_, dotk::matrix<Real> & R_);

}

}

#endif /* DOTK_QR_HPP_ */
