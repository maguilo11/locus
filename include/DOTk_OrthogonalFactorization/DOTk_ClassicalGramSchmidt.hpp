/*
 * DOTk_ClassicalGramSchmidt.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_CLASSICALGRAMSCHMIDT_HPP_
#define DOTK_CLASSICALGRAMSCHMIDT_HPP_

#include "DOTk_OrthogonalFactorization.hpp"

namespace dotk
{

template<class Type>
class matrix;

class DOTk_ClassicalGramSchmidt : public dotk::DOTk_OrthogonalFactorization
{
public:
    DOTk_ClassicalGramSchmidt();
    virtual ~DOTk_ClassicalGramSchmidt();

    void factorization(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                       std::tr1::shared_ptr<dotk::matrix<Real> > & Q_,
                       std::tr1::shared_ptr<dotk::matrix<Real> > & R_);

private:
    DOTk_ClassicalGramSchmidt(const dotk::DOTk_ClassicalGramSchmidt &);
    dotk::DOTk_ClassicalGramSchmidt & operator=(const dotk::DOTk_ClassicalGramSchmidt & rhs_);
};

}

#endif /* DOTK_CLASSICALGRAMSCHMIDT_HPP_ */
