/*
 * TRROM_MOCK_MatlabQR.cpp
 *
 *  Created on: Aug 18, 2016
 */

#include <cassert>
#include "TRROM_Basis.hpp"
#include "TRROM_SerialVector.hpp"
#include "TRROM_MOCK_MatlabQR.hpp"

namespace trrom
{

namespace mock
{

MatlabQR::MatlabQR()
{
}

MatlabQR::~MatlabQR()
{
}

trrom::types::ortho_factorization_t MatlabQR::type() const
{
    return (trrom::types::MODIFIED_GRAM_SCHMIDT_QR);
}

void MatlabQR::factorize(const std::tr1::shared_ptr<trrom::Matrix<double> > & input_,
                         std::tr1::shared_ptr<trrom::Matrix<double> > & Q_,
                         std::tr1::shared_ptr<trrom::Matrix<double> > & R_)
{
    const int num_columns = input_->getNumCols();
    trrom::SerialVector<double> x(input_->getNumRows());
    Q_.reset(new trrom::Basis<double>(x, num_columns));

    trrom::SerialVector<double> y(num_columns);
    R_.reset(new trrom::Basis<double>(x, num_columns));

    assert(Q_->getNumRows() == 4u);
    assert(Q_->getNumCols() == 2u);
    assert(R_->getNumRows() == 2u);
    assert(R_->getNumCols() == 2u);

    // Column 1
    Q_->fill(0.);
    Q_->replaceGlobalValue(0, 0, -0.188554880770777);
    Q_->replaceGlobalValue(1, 0, 0.060602976341575);
    Q_->replaceGlobalValue(2, 0, 0.973075892024672);
    Q_->replaceGlobalValue(3, 0, -0.117888271496568);
    // Column 2
    Q_->replaceGlobalValue(0, 1, 0.012359815830228);
    Q_->replaceGlobalValue(1, 1, -0.098878526641827);
    Q_->replaceGlobalValue(2, 1, -0.111238342472056);
    Q_->replaceGlobalValue(3, 1, -0.988785266418272);

    // Column 1
    R_->fill(0.);
    R_->replaceGlobalValue(0, 0, 9.410934453860165);
    // Column 2
    R_->replaceGlobalValue(0, 1, -1.734373134972489);
    R_->replaceGlobalValue(1, 1, -8.095679368799605);
}

}

}
