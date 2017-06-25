/*
 * TRROM_LinearAlgebraFactory.hpp
 *
 *  Created on: Dec 8, 2016
 *      Author: maguilo
 */

#ifndef TRROM_LINEARALGEBRAFACTORY_HPP_
#define TRROM_LINEARALGEBRAFACTORY_HPP_

#include <memory>

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class LinearAlgebraFactory
{
public:
    //! @name Constructors/destructors
    //@{
    //! LinearAlgebraFactory destructor
    virtual ~LinearAlgebraFactory()
    {
    }
    //!@}

    //! @name Pure virtual functions
    /*! Reshapes (m*n)-by-1 vector into m-by-n matrix
     * Parameters:
     *    \param In
     *          num_rows_: number of rows (m) of output matrix
     *    \param In
     *          num_columns_: number of columns (n) of output matrix
     *    \param In
     *          input_: (m*n)-by-1 vector
     *    \param Out
     *          output_: m-by-n matrix
     **/
    virtual void reshape(const int & num_rows_,
                         const int & num_columns_,
                         const std::shared_ptr<trrom::Vector<double> > & input_,
                         std::shared_ptr<trrom::Matrix<double> > & output_) = 0;
    /*! Creates a custom local vector of given length.
     * Parameters:
     *    \param In
     *          length_: vector length
     *    \param Out
     *          output_: custom local vector
     **/
    virtual void buildLocalVector(const int & length_, std::shared_ptr<trrom::Vector<double> > & output_) = 0;
    /*! Creates a m-by-n custom local matrix
     * Parameters:
     *    \param In
     *          num_rows_: number of rows (m)
     *    \param In
     *          num_columns_: number of columns (n)
     *    \param Out
     *          output_: m-by-n custom local matrix
     **/
    virtual void buildLocalMatrix(const int & num_rows_,
                                  const int & num_columns_,
                                  std::shared_ptr<trrom::Matrix<double> > & output_) = 0;
    /*! Creates a m-by-n custom dual multi-vector
     * Parameters:
     *    \param In
     *          num_vectors_: number of vectors (n)
     *    \param In
     *          vector_: vector template, dimension m is dictated by the vector's length
     *    \param Out
     *          output_: m-by-n custom dual multivector data structure
     **/
    virtual void buildMultiVector(const int & num_vectors_,
                                  const std::shared_ptr<trrom::Vector<double> > & vector_,
                                  std::shared_ptr<trrom::Matrix<double> > & output_) = 0;
};

}

#endif /* TRROM_LINEARALGEBRAFACTORY_HPP_ */
