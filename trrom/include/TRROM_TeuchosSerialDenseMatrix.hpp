/*
 * TRROM_TeuchosSerialDenseMatrix.hpp
 *
 *  Created on: Oct 19, 2016
 *      Author: maguilo
 */

#ifndef TRROM_TEUCHOSSERIALDENSEMATRIX_HPP_
#define TRROM_TEUCHOSSERIALDENSEMATRIX_HPP_

#include "TRROM_Matrix.hpp"
#include "TRROM_SerialArray.hpp"
#include "TRROM_TeuchosArray.hpp"

#include "Teuchos_SerialDenseMatrix.hpp"

namespace trrom
{

template<typename ScalarType>
class TeuchosSerialDenseMatrix : public trrom::Matrix<ScalarType>
{
public:
    TeuchosSerialDenseMatrix() :
            m_Data(new Teuchos::SerialDenseMatrix<int, ScalarType>()),
            m_Vector(new trrom::SerialArray<ScalarType>(1))
    {
    }
    TeuchosSerialDenseMatrix(int num_rows_, int num_columns_) :
            m_Data(new Teuchos::SerialDenseMatrix<int, ScalarType>(num_rows_, num_columns_)),
            m_Vector(new trrom::SerialArray<ScalarType>(1))
    {
    }
    virtual ~TeuchosSerialDenseMatrix()
    {
    }

    void fill(ScalarType value_)
    {
        m_Data->putScalar(value_);
    }
    void scale(ScalarType value_)
    {
        m_Data->scale(value_);
    }
    void insert(const trrom::Vector<ScalarType> & input_)
    {
        // TODO: RECONSIDER PURE VIRTUAL FUNCTION
    }

    void copy(const trrom::Matrix<ScalarType> & input_)
    {
        m_Data->assign(*(dynamic_cast<const trrom::TeuchosSerialDenseMatrix<ScalarType>&>(input_).m_Data));
    }
    void add(const ScalarType & alpha_, const trrom::Matrix<ScalarType> & input_)
    {
        int num_rows = input_.numRows();
        int num_columns = input_.numCols();
        assert(num_rows == this->numRows());
        assert(num_columns == this->numCols());

        for(int row_index = 0; row_index < num_rows; ++row_index)
        {
            for(int column_index = 0; column_index < num_columns; ++column_index)
            {
                m_Data->operator ()(row_index, column_index) += alpha_ * input_(row_index, column_index);
            }
        }
    }
    void gemv(bool transpose_,
              const ScalarType & alpha_,
              const trrom::Vector<ScalarType> & input_,
              const ScalarType & beta_,
              trrom::Vector<ScalarType> & output_) const
    {
        int num_rows = this->numRows();
        int num_columns = this->numCols();
        int matrix_leading_dim = std::max(1, num_rows);
        trrom::TeuchosArray<double> in(num_columns);
        in.copy(input_);
        int increments_for_in_vector = 1;
        trrom::TeuchosArray<double> out(num_rows);
        int increments_for_out_vector = 1;

        Teuchos::ETransp transpose = this->castTranspose(transpose_);

        m_Data->GEMV(transpose,
                     num_rows,
                     num_columns,
                     alpha_,
                     m_Data->values(),
                     matrix_leading_dim,
                     in.data()->getRawPtr(),
                     increments_for_in_vector,
                     beta_,
                     out.data()->getRawPtr(),
                     increments_for_out_vector);

        assert(output_.size() == out.size());
        output_.copy(out);
    }
    void gemm(const bool & transpose_A_,
              const bool & transpose_B_,
              const ScalarType & alpha_,
              const trrom::Matrix<ScalarType> & B_,
              const ScalarType & beta_,
              trrom::Matrix<ScalarType> & C_) const
    {
        int num_rows = this->numRows();
        int num_columns = this->numCols();
        int matrix_A_leading_dim = std::max(1, num_rows);
        int matrix_B_leading_dim = num_columns;
        int matrix_C_leading_dim = std::max(1, num_rows);

        trrom::TeuchosSerialDenseMatrix<ScalarType> B_matrix(B_.numRows(), B_.numCols());
        B_matrix.copy(B_);
        trrom::TeuchosSerialDenseMatrix<ScalarType> C_matrix(C_.numRows(), C_.numCols());
        C_matrix.copy(C_);

        Teuchos::ETransp transpose_A = this->castTranspose(transpose_A_);
        Teuchos::ETransp transpose_B = this->castTranspose(transpose_B_);

        m_Data->GEMM(transpose_A,
                     transpose_B,
                     num_rows,
                     num_columns,
                     num_columns,
                     alpha_,
                     m_Data->values(),
                     matrix_A_leading_dim,
                     B_matrix.data()->values(),
                     matrix_B_leading_dim,
                     beta_,
                     C_matrix.data()->values(),
                     matrix_C_leading_dim);

        C_.copy(C_matrix);
    }

    int numRows() const
    {
        int number_of_rows = m_Data->numRows();
        return (number_of_rows);
    }
    int numCols() const
    {
        int number_of_columns = m_Data->numCols();
        return (number_of_columns);
    }
    ScalarType & operator ()(const int & row_index_, const int & column_index_)
    {
        return (m_Data->operator ()(row_index_, column_index_));
    }
    const ScalarType & operator ()(const int & row_index_, const int & column_index_) const
    {
        return (m_Data->operator ()(row_index_, column_index_));
    }

    trrom::Vector<ScalarType> & vector(int index_) const
    {
        // TODO: RECONSIDER PURE VIRTUAL FUNCTION
        return (*m_Vector);
    }

    std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create() const
    {
        int rows = this->numRows();
        int cols = this->numCols();
        std::tr1::shared_ptr<trrom::Matrix<ScalarType> >
            a_copy(new trrom::TeuchosSerialDenseMatrix<ScalarType>(rows, cols));
        return (a_copy);
    }
    std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create(int nrows_, int ncols_) const
    {
        std::tr1::shared_ptr<trrom::Matrix<ScalarType> >
            a_copy(new trrom::TeuchosSerialDenseMatrix<ScalarType>(nrows_, ncols_));
        return (a_copy);
    }
    std::tr1::shared_ptr<Teuchos::SerialDenseMatrix<int, ScalarType> > & data()
    {
        return (m_Data);
    }

private:
    Teuchos::ETransp castTranspose(bool transpose_) const
    {
        Teuchos::ETransp transpose;
        if(transpose_ == true)
        {
            transpose = Teuchos::ETransp::TRANS;
        }
        else
        {

            transpose = Teuchos::ETransp::NO_TRANS;
        }
        return (transpose);
    }

private:
    std::tr1::shared_ptr< Teuchos::SerialDenseMatrix<int, ScalarType> > m_Data;
    std::tr1::shared_ptr< trrom::SerialArray<ScalarType> > m_Vector;

private:
    TeuchosSerialDenseMatrix(const trrom::TeuchosSerialDenseMatrix<ScalarType> &);
    trrom::TeuchosSerialDenseMatrix<ScalarType> & operator=(const trrom::TeuchosSerialDenseMatrix<ScalarType> & rhs_);
};

}

#endif /* TRROM_TEUCHOSSERIALDENSEMATRIX_HPP_ */
