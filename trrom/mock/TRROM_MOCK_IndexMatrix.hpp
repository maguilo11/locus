/*
 * TRROM_MOCK_IndexMatrix.hpp
 *
 *  Created on: Aug 18, 2016
 */

#ifndef TRROM_MOCK_INDEXMATRIX_HPP_
#define TRROM_MOCK_INDEXMATRIX_HPP_

#include "TRROM_Basis.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

namespace mock
{

template<typename ScalarType>
class IndexMatrix : public trrom::Matrix<ScalarType>
{
public:
    IndexMatrix(const trrom::Vector<ScalarType> & input_, const int num_columns_) :
            m_Matrix(new trrom::Basis<ScalarType>(input_, num_columns_))
    {
    }
    virtual ~IndexMatrix()
    {
    }

    int getNumRows() const
    {
        return (m_Matrix->getNumRows());
    }
    int getNumCols() const
    {
        return (m_Matrix->getNumCols());
    }
    void fill(ScalarType value_)
    {
        m_Matrix->fill(value_);
    }
    void scale(ScalarType value_)
    {
        m_Matrix->scale(value_);
    }
    void insert(const trrom::Vector<ScalarType> & input_)
    {
        m_Matrix->insert(input_);
    }
    void update(const ScalarType & alpha_, const trrom::Matrix<ScalarType> & input_, const ScalarType & beta_)
    {
        m_Matrix->update(alpha_, input_, beta_);
    }
    void gemv(bool transpose_,
              const ScalarType & alpha_,
              const trrom::Vector<ScalarType> & input_,
              const ScalarType & beta_,
              trrom::Vector<ScalarType> & output_) const
    {
        m_Matrix->gemv(transpose_, alpha_, input_, beta_, output_);
    }
    void gemm(const bool & transpose_A_,
              const bool & transpose_B_,
              const ScalarType & alpha_,
              const trrom::Matrix<ScalarType> & B_,
              const ScalarType & beta_,
              trrom::Matrix<ScalarType> & C_) const
    {
        m_Matrix->gemm(transpose_A_, transpose_B_, alpha_, B_, beta_, C_);
    }
    double getGlobalValue(const int & global_row_index_, const int & global_column_index_) const
    {
        return (m_Matrix->getGlobalValue(global_row_index_, global_column_index_));
    }
    void replaceGlobalValue(const int & global_row_index_, const int & global_column_index_, const ScalarType & value_)
    {
        m_Matrix->replaceGlobalValue(global_row_index_, global_column_index_, value_);
    }
    std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create(int nrows_ = 0, int ncols_ = 0) const
    {
        assert(nrows_ >= 0);
        assert(ncols_ >= 0);
        std::tr1::shared_ptr<trrom::mock::IndexMatrix<ScalarType> > this_copy;
        if((nrows_ > 0) && (ncols_ > 0))
        {
            trrom::SerialVector<ScalarType> vector(nrows_);
            this_copy.reset(new trrom::mock::IndexMatrix<ScalarType>(vector, ncols_));
        }
        else
        {
            int num_rows = this->getNumRows();
            int num_cols = this->getNumCols();
            trrom::SerialVector<ScalarType> vector(num_rows);
            std::tr1::shared_ptr<trrom::mock::IndexMatrix<ScalarType> >
                matrix(new trrom::mock::IndexMatrix<ScalarType>(vector, num_cols));
        }
        return (this_copy);
    }

    trrom::Vector<ScalarType> & vector(int index_) const
    {
        return (m_Matrix->vector(index_));
    }
    ScalarType trace() const
    {
        return (m_Matrix->trace());
    }

private:
    std::tr1::shared_ptr<trrom::Basis<ScalarType> > m_Matrix;

private:
    IndexMatrix(const trrom::mock::IndexMatrix<ScalarType> &);
    trrom::mock::IndexMatrix<ScalarType> & operator=(const trrom::mock::IndexMatrix<ScalarType> &);
};

}

}

#endif /* TRROM_MOCK_INDEXMATRIX_HPP_ */
