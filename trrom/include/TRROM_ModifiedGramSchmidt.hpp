/*
 * TRROM_ModifiedGramSchmidt.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_MODIFIEDGRAMSCHMIDT_HPP_
#define TRROM_MODIFIEDGRAMSCHMIDT_HPP_

#include "TRROM_OrthogonalFactorization.hpp"

namespace trrom
{

class ModifiedGramSchmidt : public trrom::OrthogonalFactorization
{
public:
    ModifiedGramSchmidt();
    virtual ~ModifiedGramSchmidt();

    trrom::types::ortho_factorization_t type() const;
    virtual void factorize(const trrom::Matrix<double> & input_, trrom::Matrix<double> & Q_, trrom::Matrix<double> & R_);

private:
    ModifiedGramSchmidt(const trrom::ModifiedGramSchmidt &);
    trrom::ModifiedGramSchmidt & operator=(const trrom::ModifiedGramSchmidt &);
};

}

#endif /* TRROM_MODIFIEDGRAMSCHMIDT_HPP_ */
