/*
 * DOTk_ModifiedGramSchmidt.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MODIFIEDGRAMSCHMIDT_HPP_
#define DOTK_MODIFIEDGRAMSCHMIDT_HPP_

#include "DOTk_OrthogonalFactorization.hpp"

namespace dotk
{

template<class Type>
class matrix;

class DOTk_ModifiedGramSchmidt : public dotk::DOTk_OrthogonalFactorization
{
public:
    DOTk_ModifiedGramSchmidt();
    virtual ~DOTk_ModifiedGramSchmidt();

    void factorization(const std::shared_ptr<dotk::matrix<Real> > & input_,
                       std::shared_ptr<dotk::matrix<Real> > & Q_,
                       std::shared_ptr<dotk::matrix<Real> > & R_);

private:
    DOTk_ModifiedGramSchmidt(const dotk::DOTk_ModifiedGramSchmidt &);
    dotk::DOTk_ModifiedGramSchmidt & operator=(const dotk::DOTk_ModifiedGramSchmidt & rhs_);
};

}

#endif /* DOTK_MODIFIEDGRAMSCHMIDT_HPP_ */
