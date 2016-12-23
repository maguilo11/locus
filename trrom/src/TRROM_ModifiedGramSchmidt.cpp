/*
 * TRROM_ModifiedGramSchmidt.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include <cassert>

#include "TRROM_Matrix.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_ModifiedGramSchmidt.hpp"
#include "TRROM_LinearAlgebraFactory.hpp"

namespace trrom
{

ModifiedGramSchmidt::ModifiedGramSchmidt(const std::tr1::shared_ptr<trrom::LinearAlgebraFactory> & factory_) :
        m_Factory(factory_)
{
}

ModifiedGramSchmidt::~ModifiedGramSchmidt()
{
}

trrom::types::ortho_factorization_t ModifiedGramSchmidt::type() const
{
    return (trrom::types::MODIFIED_GRAM_SCHMIDT_QR);
}

void ModifiedGramSchmidt::factorize(const std::tr1::shared_ptr<trrom::Matrix<double> > & input_,
                                    std::tr1::shared_ptr<trrom::Matrix<double> > & Q_,
                                    std::tr1::shared_ptr<trrom::Matrix<double> > & R_)
{
    assert(input_->getNumRows() > 0);
    assert(input_->getNumCols() > 0);

    // allocate output data
    Q_ = input_->create();
    int num_columns = input_->getNumCols();
    m_Factory->buildLocalMatrix(num_columns, num_columns, R_);

    // Perform orthogonal factorization
    R_->fill(0.);
    Q_->update(1., *input_, 0.);
    num_columns = Q_->getNumCols();
    for(int index_i = 0; index_i < num_columns; ++index_i)
    {
        double value = Q_->vector(index_i)->norm();
        R_->replaceGlobalValue(index_i, index_i, value);
        value = static_cast<double>(1.) / value;
        Q_->vector(index_i)->scale(value);
        for(int index_j = index_i + 1; index_j < num_columns; ++index_j)
        {
            value = Q_->vector(index_i)->dot(*Q_->vector(index_j));
            R_->replaceGlobalValue(index_i, index_j, value);
            Q_->vector(index_j)->update(-value, *Q_->vector(index_i), 1.);
        }
    }
}

}
