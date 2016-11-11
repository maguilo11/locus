/*
 * TRROM_Basis.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_BASIS_HPP_
#define TRROM_BASIS_HPP_

#include "TRROM_Matrix.hpp"
#include "TRROM_SerialVector.hpp"

namespace trrom
{

template<typename ScalarType>
class Basis : public trrom::Matrix<ScalarType>
{
public:
    Basis() :
            m_NumSnapShots(0),
            m_Data()
    {
    }
    Basis(const trrom::Vector<ScalarType> & input_, const int num_columns_) :
            m_NumSnapShots(0),
            m_Data()
    {
        this->initialize(input_, num_columns_);
    }
    ~Basis()
    {
    }

    int numRows() const
    {
        int num_rows = m_Data[0]->size();
        return (num_rows);
    }
    int numCols() const
    {
        int num_cols = m_Data.size();
        return (num_cols);
    }
    int num_snapshots() const
    {
        return (m_NumSnapShots);
    }

    ScalarType norm() const
    {
        ScalarType value = 0.;
        for(int column = 0; column < this->numCols(); ++column)
        {
            value += m_Data[column]->dot(*m_Data[column]);
        }
        value = std::pow(value, 0.5);
        return (value);
    }
    void fill(ScalarType value_)
    {
        for(int column = 0; column < this->numCols(); ++column)
        {
            m_Data[column]->fill(value_);
        }
    }
    void scale(ScalarType alpha_)
    {
        for(int column = 0; column < this->numCols(); ++column)
        {
            m_Data[column]->scale(alpha_);
        }
    }
    void insert(const trrom::Vector<ScalarType> & input_)
    {
        int this_vector_dimension = m_Data.size();
        if(m_NumSnapShots < this_vector_dimension)
        {
            m_Data[m_NumSnapShots]->copy(input_);
        }
        else
        {
            m_Data.push_back(input_.create());
            m_Data[m_NumSnapShots]->copy(input_);
        }
        m_NumSnapShots += 1;
    }

    void copy(const trrom::Vector<ScalarType> & input_)
    {
        assert(this->numCols() * this->numRows() == input_.size());

        for(int column = 0; column < this->numCols(); ++column)
        {
            for(int row = 0; row < this->numRows(); ++row)
            {
                int index = (this->numRows() * column) + row;
                m_Data[column]->operator[](row) = input_[index];
            }
        }
    }
    void copy(const trrom::Matrix<ScalarType> & input_)
    {
        assert(this->numCols() == input_.numCols());
        assert(this->numRows() == input_.numRows());

        for(int column = 0; column < input_.numCols(); ++column)
        {
            for(int row = 0; row < input_.numRows(); ++row)
            {
                m_Data[column]->operator[](row) = input_(row, column);
            }
        }
    }
    void add(const ScalarType & alpha_, const trrom::Matrix<ScalarType> & input_)
    {
        assert(this->numCols() == input_.numCols());
        assert(this->numRows() == input_.numRows());

        for(int column = 0; column < input_.numCols(); ++column)
        {
            for(int row = 0; row < input_.numRows(); ++row)
            {
                m_Data[column]->operator[](row) = alpha_ * input_(row, column) + m_Data[column]->operator[](row);
            }
        }
    }
    void gemv(bool transpose_,
                      const ScalarType & alpha_,
                      const trrom::Vector<ScalarType> & input_,
                      const ScalarType & beta_,
                      trrom::Vector<ScalarType> & output_) const
    {
        if(transpose_ == false)
        {
            assert(this->numCols() == input_.size());
            assert(this->numRows() == output_.size());

            int i;
            for(i = 0; i < output_.size(); ++i)
            {
                output_[i] = beta_ * output_[i];
            }

            int row, column;
            ScalarType value, value_to_add;

            for(column = 0; column < input_.size(); ++column)
            {
                for(row = 0; row < output_.size(); ++row)
                {
                    value = alpha_ * m_Data[column]->operator[](row) * input_.operator[](column);
                    value_to_add = output_.operator[](row) + value;
                    output_.operator[](row) = value_to_add;
                }
            }
        }
        else
        {
            assert(this->numRows() == input_.size());
            assert(this->numCols() == output_.size());

            int row, column;
            ScalarType value, sum, beta_times_output;

            for(column = 0; column < output_.size(); ++column)
            {
                sum = 0.;
                for(row = 0; row < input_.size(); ++row)
                {
                    value = m_Data[column]->operator[](row) * input_.operator[](row);
                    sum += alpha_ * value;
                }
                beta_times_output = beta_ * output_.operator[](column);
                output_.operator[](column) = sum + beta_times_output;
            }
        }
    }
    void gemm(const bool & transpose_A_,
                      const bool & transpose_B_,
                      const ScalarType & alpha_,
                      const trrom::Matrix<ScalarType> & B_,
                      const ScalarType & beta_,
                      trrom::Matrix<ScalarType> & C_) const
    {
        // Quick return if possible
        if(alpha_ == static_cast<ScalarType>(0.))
        {
            if(beta_ == static_cast<ScalarType>(0.))
            {
                C_.fill(0.);
            }
            else
            {
                C_.scale(beta_);
            }
            return;
        }
        // Scale Output Matrix if Necessary
        if(beta_ != static_cast<ScalarType>(0.))
        {
            C_.scale(beta_);
        }
        else
        {
            C_.fill(0.);
        }
        // Start Operations
        if(transpose_B_ == false)
        {
            this->gemmB(transpose_A_, alpha_, B_, C_);
        }
        else
        {
            this->gemmBt(transpose_A_, alpha_, B_, C_);
        }
    }

    trrom::Vector<ScalarType> & vector(int index_) const
    {
        return (*m_Data[index_]);
    }
    ScalarType & operator ()(const int & row_index_, const int & column_index_)
    {
        assert(row_index_ <= this->numRows() - 1);
        assert(column_index_ <= this->numCols() - 1);
        return (m_Data[column_index_]->operator[](row_index_));
    }
    // Operator overloads the parenthesis operator
    const ScalarType & operator ()(const int & row_index_, const int & column_index_) const
    {
        assert(row_index_ <= this->numRows() - 1);
        assert(column_index_ <= this->numCols() - 1);
        return (m_Data[column_index_]->operator[](row_index_));
    }
    std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create() const
    {
        int num_rows = this->numRows();
        int num_cols = this->numCols();
        trrom::SerialVector<ScalarType> vector(num_rows);
        std::tr1::shared_ptr<trrom::Basis<ScalarType> > matrix(new trrom::Basis<ScalarType>(vector, num_cols));
        return (matrix);
    }
    std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create(int nrows_, int ncols_) const
    {
        trrom::SerialVector<ScalarType> vector(nrows_);
        std::tr1::shared_ptr<trrom::Basis<ScalarType> > matrix(new trrom::Basis<ScalarType>(vector, ncols_));
        return (matrix);
    }

    int snapshots() const
    {
        return (m_NumSnapShots);
    }

private:
    void initialize(const trrom::Vector<ScalarType> & input_, int num_columns_)
    {
        m_Data.resize(num_columns_);
        for(int index = 0; index < num_columns_; ++index)
        {
            m_Data[index] = input_.create();
        }
    }
    void gemmB(const bool & transpose_A_,
               const ScalarType & alpha_,
               const trrom::Matrix<ScalarType> & B_,
               trrom::Matrix<ScalarType> & C_) const
    {
        if(transpose_A_ == false)
        {
            // C = (A)(B)
            assert(this->numCols() == B_.numRows());
            assert(C_.numRows() == this->numRows());
            assert(C_.numCols() == B_.numCols());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->numRows();
            int num_cols_A = this->numCols();
            int num_cols_B = B_.numCols();

            for(j = 0; j < num_cols_B; ++j)
            {
                for(k = 0; k < num_cols_A; ++k)
                {
                    for(i = 0; i < num_rows_A; ++i)
                    {
                        value = alpha_ * m_Data[k]->operator[](i) * B_(k, j);
                        C_(i, j) += value;
                    }
                }
            }
        }
        else
        {
            // C = (A^t)(B)
            assert(this->numRows() == B_.numRows());
            assert(C_.numRows() == this->numCols());
            assert(C_.numCols() == B_.numCols());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->numRows();
            int num_cols_A = this->numCols();
            int num_cols_B = B_.numCols();

            for(j = 0; j < num_cols_B; ++j)
            {
                for(i = 0; i < num_cols_A; ++i)
                {
                    for(k = 0; k < num_rows_A; ++k)
                    {
                        value = alpha_ * m_Data[i]->operator[](k) * B_(k, j);
                        C_(i, j) += value;
                    }
                }
            }
        }
    }
    void gemmBt(const bool & transpose_A_,
                const ScalarType & alpha_,
                const trrom::Matrix<ScalarType> & B_,
                trrom::Matrix<ScalarType> & C_) const
    {
        if(transpose_A_ == false)
        {
            // C = (A)(B^t)
            assert(this->numCols() == B_.numCols());
            assert(C_.numRows() == this->numRows());
            assert(C_.numCols() == B_.numRows());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->numRows();
            int num_cols_A = this->numCols();
            int num_rows_B = B_.numRows();

            for(k = 0; k < num_cols_A; ++k)
            {
                for(j = 0; j < num_rows_B; ++j)
                {
                    for(i = 0; i < num_rows_A; ++i)
                    {
                        value = alpha_ * m_Data[k]->operator[](i) * B_(j, k);
                        C_(i, j) += value;
                    }
                }
            }
        }
        else
        {
            // C = (A^t)(B^t)
            assert(this->numRows() == B_.numCols());
            assert(C_.numRows() == this->numCols());
            assert(C_.numCols() == B_.numRows());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->numRows();
            int num_cols_A = this->numCols();
            int num_rows_B = B_.numRows();

            for(i = 0; i < num_cols_A; ++i)
            {
                for(k = 0; k < num_rows_A; ++k)
                {
                    for(j = 0; j < num_rows_B; ++j)
                    {
                        value = alpha_ * m_Data[i]->operator[](k) * B_(j, k);
                        C_(i, j) += value;
                    }
                }
            }
        }
    }

private:
    int m_NumSnapShots;
    std::vector<std::tr1::shared_ptr<trrom::Vector<ScalarType> > > m_Data;

private:
    Basis(const trrom::Basis<ScalarType> &);
    trrom::Basis<ScalarType> & operator=(const trrom::Basis<ScalarType> &);
};

}

#endif /* TRROM_BASIS_HPP_ */
