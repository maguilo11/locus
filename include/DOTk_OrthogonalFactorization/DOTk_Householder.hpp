/*
 * DOTk_Householder.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_HOUSEHOLDER_HPP_
#define DOTK_HOUSEHOLDER_HPP_

#include "DOTk_OrthogonalFactorization.hpp"

namespace dotk
{

template<class Type>
class matrix;

class DOTk_Householder : public dotk::DOTk_OrthogonalFactorization
{
public:
    DOTk_Householder();
    virtual ~DOTk_Householder();

    void factorization(const std::shared_ptr<dotk::matrix<Real> > & input_,
                       std::shared_ptr<dotk::matrix<Real> > & Q_,
                       std::shared_ptr<dotk::matrix<Real> > & R_);

private:
    DOTk_Householder(const dotk::DOTk_Householder &);
    dotk::DOTk_Householder & operator=(const dotk::DOTk_Householder & rhs_);
};

}

#endif /* DOTK_HOUSEHOLDER_HPP_ */
