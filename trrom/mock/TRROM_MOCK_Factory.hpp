/*
 * TRROM_MOCK_Factory.hpp
 *
 *  Created on: Dec 20, 2016
 *      Author: maguilo
 */

#ifndef TRROM_MOCK_FACTORY_HPP_
#define TRROM_MOCK_FACTORY_HPP_

#include <memory>

#include "TRROM_LinearAlgebraFactory.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;
template<typename ScalarType>
class Matrix;

namespace mock
{

class Factory : public trrom::LinearAlgebraFactory
{
public:
    Factory();
    virtual ~Factory();

    void reshape(const int & num_rows_,
                 const int & num_columns_,
                 const std::shared_ptr<trrom::Vector<double> > & input_,
                 std::shared_ptr<trrom::Matrix<double> > & output_);
    void buildLocalVector(const int & length_, std::shared_ptr<trrom::Vector<double> > & output_);
    void buildLocalMatrix(const int & num_rows_,
                          const int & num_columns_,
                          std::shared_ptr<trrom::Matrix<double> > & output_);
    void buildMultiVector(const int & num_vectors_,
                          const std::shared_ptr<trrom::Vector<double> > & vector_,
                          std::shared_ptr<trrom::Matrix<double> > & output_);

private:
    Factory(const trrom::mock::Factory &);
    trrom::mock::Factory & operator=(const trrom::mock::Factory &);
};
// end Factory class

}
// end mock namespace

}
// end trrom namespace

#endif /* TRROM_MOCK_FACTORY_HPP_ */
