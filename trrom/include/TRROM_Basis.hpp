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

    int getNumRows() const
    {
        int num_rows = m_Data[0]->size();
        return (num_rows);
    }
    int getNumCols() const
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
        for(int column = 0; column < this->getNumCols(); ++column)
        {
            value += m_Data[column]->dot(*m_Data[column]);
        }
        value = std::pow(value, 0.5);
        return (value);
    }
    void fill(ScalarType value_)
    {
        for(int column = 0; column < this->getNumCols(); ++column)
        {
            m_Data[column]->fill(value_);
        }
    }
    void scale(ScalarType alpha_)
    {
        for(int column = 0; column < this->getNumCols(); ++column)
        {
            m_Data[column]->scale(alpha_);
        }
    }
    void insert(const trrom::Vector<ScalarType> & input_, int index_ = 0)
    {
        int this_vector_dimension = m_Data.size();
        if(m_NumSnapShots < this_vector_dimension)
        {
            m_Data[m_NumSnapShots]->update(1., input_, 0.);
        }
        else
        {
            m_Data.push_back(input_.create());
            m_Data[m_NumSnapShots]->update(1., input_, 0.);
        }
        m_NumSnapShots += 1;
    }

    void copy(const trrom::Vector<ScalarType> & input_)
    {
        assert(this->getNumCols() * this->getNumRows() == input_.size());

        for(int column = 0; column < this->getNumCols(); ++column)
        {
            for(int row = 0; row < this->getNumRows(); ++row)
            {
                int index = (this->getNumRows() * column) + row;
                m_Data[column]->operator[](row) = input_[index];
            }
        }
    }
    void update(const ScalarType & alpha_, const trrom::Matrix<ScalarType> & input_, const ScalarType & beta_)
    {
        assert(this->getNumCols() == input_.getNumCols());
        assert(this->getNumRows() == input_.getNumRows());

        for(int column = 0; column < input_.getNumCols(); ++column)
        {
            for(int row = 0; row < input_.getNumRows(); ++row)
            {
                m_Data[column]->operator[](row) = alpha_ * input_(row, column)
                                                  + beta_ * m_Data[column]->operator[](row);
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
            assert(this->getNumCols() == input_.size());
            assert(this->getNumRows() == output_.size());

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
            assert(this->getNumRows() == input_.size());
            assert(this->getNumCols() == output_.size());

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
    ScalarType & operator ()(int my_row_index_, int my_column_index_)
    {
        return (m_Data[my_column_index_]->operator[](my_row_index_));
    }
    const ScalarType & operator ()(int my_row_index_, int my_column_index_) const
    {
        return (m_Data[my_column_index_]->operator[](my_row_index_));
    }
    void replaceGlobalValue(const int & global_row_index_, const int & global_column_index_, const ScalarType & value_)
    {
        m_Data[global_column_index_]->operator[](global_row_index_) = value_;
    }
    const std::shared_ptr<trrom::Vector<ScalarType> > & vector(int index_) const
    {
        return (m_Data[index_]);
    }
    std::shared_ptr<trrom::Matrix<ScalarType> > create(int num_rows_, int num_cols_) const
    {
        assert(num_rows_ >= 0);
        assert(num_cols_ >= 0);
        std::shared_ptr<trrom::Basis<ScalarType> > this_copy;
        if((num_rows_ > 0) && (num_cols_ > 0))
        {
            trrom::SerialVector<ScalarType> vector(num_rows_);
            this_copy.reset(new trrom::Basis<ScalarType>(vector, num_cols_));
        }
        else
        {
            int num_rows = this->getNumRows();
            int num_cols = this->getNumCols();
            trrom::SerialVector<ScalarType> vector(num_rows);
            this_copy.reset(new trrom::Basis<ScalarType>(vector, num_cols));
        }
        return (this_copy);
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
            assert(this->getNumCols() == B_.getNumRows());
            assert(C_.getNumRows() == this->getNumRows());
            assert(C_.getNumCols() == B_.getNumCols());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->getNumRows();
            int num_cols_A = this->getNumCols();
            int num_cols_B = B_.getNumCols();

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
            assert(this->getNumRows() == B_.getNumRows());
            assert(C_.getNumRows() == this->getNumCols());
            assert(C_.getNumCols() == B_.getNumCols());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->getNumRows();
            int num_cols_A = this->getNumCols();
            int num_cols_B = B_.getNumCols();

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
            assert(this->getNumCols() == B_.getNumCols());
            assert(C_.getNumRows() == this->getNumRows());
            assert(C_.getNumCols() == B_.getNumRows());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->getNumRows();
            int num_cols_A = this->getNumCols();
            int num_rows_B = B_.getNumRows();

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
            assert(this->getNumRows() == B_.getNumCols());
            assert(C_.getNumRows() == this->getNumCols());
            assert(C_.getNumCols() == B_.getNumRows());

            ScalarType value = 0.;
            int i, j, k;
            int num_rows_A = this->getNumRows();
            int num_cols_A = this->getNumCols();
            int num_rows_B = B_.getNumRows();

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
    std::vector<std::shared_ptr<trrom::Vector<ScalarType> > > m_Data;

private:
    Basis(const trrom::Basis<ScalarType> &);
    trrom::Basis<ScalarType> & operator=(const trrom::Basis<ScalarType> &);
};

}

#endif /* TRROM_BASIS_HPP_ */
