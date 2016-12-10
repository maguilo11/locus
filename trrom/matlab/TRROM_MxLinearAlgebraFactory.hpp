/*
 * TRROM_MxLinearAlgebraFactory.hpp
 *
 *  Created on: Dec 8, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MXLINEARALGEBRAFACTORY_HPP_
#define TRROM_MXLINEARALGEBRAFACTORY_HPP_

#include "TRROM_LinearAlgebraFactory.hpp"

namespace trrom
{

template<typename ScalarType>
class Matrix;
template<typename ScalarType>
class Vector;

class MxLinearAlgebraFactory : public trrom::LinearAlgebraFactory
{
public:
    //! @name Constructors/destructors
    //@{
    /*!
     * Creates a MxLinearAlgebraFactory object
     * \return Reference to MxLinearAlgebraFactory.
     *
     **/
    MxLinearAlgebraFactory();
    //! LinearAlgebraFactory destructor
    virtual ~MxLinearAlgebraFactory();
    //!@}

    //! @name Functions
    /*! MEX interface that reshapes (m*n)-by-1 vector into m-by-n matrix
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
    void reshape(const int & num_rows_,
                 const int & num_columns_,
                 const std::tr1::shared_ptr<trrom::Vector<double> > & input_,
                 std::tr1::shared_ptr<trrom::Matrix<double> > & output_);
    /*! MEX interface that creates a local vector of type MxVector.
    * Parameters:
    *    \param In
    *          length_: vector length
    *    \param Out
    *          output_: MxVector vector
    **/
    void buildLocalVector(const int & length_, std::tr1::shared_ptr<trrom::Vector<double> > & output_);
    /*! Creates a m-by-n local matrix of type MxMatrix
    * Parameters:
    *    \param In
    *          num_rows_: number of rows (m)
    *    \param In
    *          num_columns_: number of columns (n)
    *    \param Out
    *          output_: m-by-n MxMatrix
    **/
    void buildLocalMatrix(const int & num_rows_,
                          const int & num_columns_,
                          std::tr1::shared_ptr<trrom::Matrix<double> > & output_);
    /*! Creates a m-by-n multi-matrix of type MxMatrix
    * Parameters:
    *    \param In
    *          num_vectors_: number of vectors (n)
     *    \param In
     *          vector_: vector template, dimension m is dictated by the vector's length
    *    \param Out
    *          output_: m-by-n MxMatrix
    **/
    void buildMultiVector(const int & num_vectors_,
                          const std::tr1::shared_ptr<trrom::Vector<double> > & vector_,
                          std::tr1::shared_ptr<trrom::Matrix<double> > & output_);

private:
    MxLinearAlgebraFactory(const trrom::MxLinearAlgebraFactory &);
    trrom::MxLinearAlgebraFactory & operator=(const trrom::MxLinearAlgebraFactory &);
};

}

#endif /* TRROM_MXLINEARALGEBRAFACTORY_HPP_ */
