/*
 * TRROM_ModifiedGramSchmidt.cpp
 *
 *  Created on: Aug 17, 2016
 */

#include <cassert>

#include "TRROM_Matrix.hpp"
#include "TRROM_Vector.hpp"
#include "TRROM_ModifiedGramSchmidt.hpp"

namespace trrom
{

ModifiedGramSchmidt::ModifiedGramSchmidt()
{
}

ModifiedGramSchmidt::~ModifiedGramSchmidt()
{
}

trrom::types::ortho_factorization_t ModifiedGramSchmidt::type() const
{
    return (trrom::types::MODIFIED_GRAM_SCHMIDT_QR);
}

void ModifiedGramSchmidt::factorize(const trrom::Matrix<double> & input_,
                                    trrom::Matrix<double> & Q_,
                                    trrom::Matrix<double> & R_)
{
    assert(R_.getNumCols() > 0);
    assert(R_.getNumCols() > 0);
    assert(Q_.getNumRows() == input_.getNumRows());
    assert(Q_.getNumCols() == input_.getNumCols());

    R_.fill(0.);
    Q_.update(1., input_, 0.);
    int num_columns = Q_.getNumCols();
    for(int index_i = 0; index_i < num_columns; ++index_i)
    {
        double value = Q_.vector(index_i).norm();
        R_.replaceGlobalValue(index_i, index_i, value);
        value = static_cast<double>(1.) / value;
        Q_.vector(index_i).scale(value);
        for(int index_j = index_i + 1; index_j < num_columns; ++index_j)
        {
            value = Q_.vector(index_i).dot(Q_.vector(index_j));
            R_.replaceGlobalValue(index_i, index_j, value);
            Q_.vector(index_j).update(-value, Q_.vector(index_i), 1.);
        }
    }
}

}
