/*
 * TRROM_MOCK_Factory.cpp
 *
 *  Created on: Dec 20, 2016
 *      Author: maguilo
 */

#include <cstdio>
#include <sstream>
#include <cstdlib>

#include "TRROM_Basis.hpp"
#include "TRROM_SerialVector.hpp"
#include "TRROM_MOCK_Factory.hpp"

namespace trrom
{

namespace mock
{

Factory::Factory()
{
}

Factory::~Factory()
{
}

void Factory::reshape(const int & num_rows_,
                      const int & num_columns_,
                      const std::shared_ptr<trrom::Vector<double> > & input_,
                      std::shared_ptr<trrom::Matrix<double> > & output_)
{
    std::ostringstream error;
    error << "\nERROR IN: " << __FILE__ << ", LINE: " << __LINE__ << ", FUNCTION: " << __FUNCTION__
            << ", MESSAGE: RESHAPE FUNCTION IS NOT DEFINED.\n";
    std::perror(error.str().c_str());
    std::abort();
}

void Factory::buildLocalVector(const int & length_, std::shared_ptr<trrom::Vector<double> > & output_)
{
    output_.reset(new trrom::SerialVector<double>(length_));
}

void Factory::buildLocalMatrix(const int & num_rows_,
                               const int & num_columns_,
                               std::shared_ptr<trrom::Matrix<double> > & output_)
{
    trrom::SerialVector<double> x(num_rows_);
    output_.reset(new trrom::Basis<double>(x, num_columns_));
}

void Factory::buildMultiVector(const int & num_vectors_,
                               const std::shared_ptr<trrom::Vector<double> > & vector_,
                               std::shared_ptr<trrom::Matrix<double> > & output_)
{
    output_.reset(new trrom::Basis<double>(*vector_, num_vectors_));
}

}

}
