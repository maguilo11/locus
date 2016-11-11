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

    ScalarType trace() const
    {
        return (m_Matrix->trace());
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

    void copy(const trrom::Matrix<ScalarType> & input_)
    {
        m_Matrix->copy(input_);
    }
    void add(const ScalarType & alpha_, const trrom::Matrix<ScalarType> & input_)
    {
        m_Matrix->add(alpha_, input_);
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

    int numRows() const
    {
        return (m_Matrix->numRows());
    }
    int numCols() const
    {
        return (m_Matrix->numCols());
    }
    ScalarType & operator ()(const int & row_index_, const int & column_index_)
    {
        return (m_Matrix->operator ()(row_index_, column_index_));
    }
    const ScalarType & operator ()(const int & row_index_, const int & column_index_) const
    {
        return (m_Matrix->operator ()(row_index_, column_index_));
    }

    trrom::Vector<ScalarType> & vector(int index_) const
    {
        return (m_Matrix->vector(index_));
    }
    std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create() const
    {
        int num_rows = this->numRows();
        int num_cols = this->numCols();
        trrom::SerialVector<ScalarType> vector(num_rows);
        std::tr1::shared_ptr<trrom::mock::IndexMatrix<ScalarType> > matrix(new trrom::mock::IndexMatrix<ScalarType>(vector, num_cols));
        return (matrix);
    }
    std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create(int nrows_, int ncols_) const
    {
        trrom::SerialVector<ScalarType> vector(nrows_);
        std::tr1::shared_ptr<trrom::mock::IndexMatrix<ScalarType> > matrix(new trrom::mock::IndexMatrix<ScalarType>(vector, ncols_));
        return (matrix);
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
