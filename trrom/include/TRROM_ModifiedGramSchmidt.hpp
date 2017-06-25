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

class LinearAlgebraFactory;

class ModifiedGramSchmidt : public trrom::OrthogonalFactorization
{
public:
    explicit ModifiedGramSchmidt(const std::shared_ptr<trrom::LinearAlgebraFactory> & factory_);
    virtual ~ModifiedGramSchmidt();

    trrom::types::ortho_factorization_t type() const;
    void factorize(const std::shared_ptr<trrom::Matrix<double> > & input_,
                   std::shared_ptr<trrom::Matrix<double> > & Q_,
                   std::shared_ptr<trrom::Matrix<double> > & R_);

private:
    std::shared_ptr<trrom::LinearAlgebraFactory> m_Factory;

private:
    ModifiedGramSchmidt(const trrom::ModifiedGramSchmidt &);
    trrom::ModifiedGramSchmidt & operator=(const trrom::ModifiedGramSchmidt &);
};

}

#endif /* TRROM_MODIFIEDGRAMSCHMIDT_HPP_ */
