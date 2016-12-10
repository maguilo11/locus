/*
 * TRROM_Matrix.hpp
 *
 *  Created on: Aug 17, 2016
 */

#ifndef TRROM_MATRIX_HPP_
#define TRROM_MATRIX_HPP_

#include <cstddef>
#include <tr1/memory>

namespace trrom
{

template<typename ScalarType>
class Vector;

template<typename ScalarType>
class Matrix
{
public:
    virtual ~Matrix()
    {
    }

    virtual int getNumRows() const = 0;
    virtual int getNumCols() const = 0;
    virtual void fill(ScalarType value_) = 0;
    virtual void scale(ScalarType value_) = 0;
    virtual void update(const ScalarType & alpha_,
                        const trrom::Matrix<ScalarType> & input_,
                        const ScalarType & beta_) = 0;
    virtual void gemv(bool transpose_,
                      const ScalarType & alpha_,
                      const trrom::Vector<ScalarType> & input_,
                      const ScalarType & beta_,
                      trrom::Vector<ScalarType> & output_) const = 0;
    virtual void gemm(const bool & transpose_A_,
                      const bool & transpose_B_,
                      const ScalarType & alpha_,
                      const trrom::Matrix<ScalarType> & B_,
                      const ScalarType & beta_,
                      trrom::Matrix<ScalarType> & C_) const = 0;
    virtual void replaceGlobalValue(const int & global_row_index_,
                                    const int & global_column_index_,
                                    const ScalarType & value_) = 0;
    virtual ScalarType & operator ()(int my_row_index_, int my_column_index_) = 0;
    virtual const ScalarType & operator ()(int my_row_index_, int my_column_index_) const = 0;

    virtual void insert(const trrom::Vector<ScalarType> & input_, int index_ = 0) = 0;
    virtual const std::tr1::shared_ptr<trrom::Vector<ScalarType> > & vector(int index_) const = 0;
    virtual std::tr1::shared_ptr<trrom::Matrix<ScalarType> > create(int nrows_ = 0, int ncols_ = 0) const = 0;
};

}

#endif /* TRROM_MATRIX_HPP_ */
