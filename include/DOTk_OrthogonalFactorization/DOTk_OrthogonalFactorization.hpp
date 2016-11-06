/*
 * DOTk_OrthogonalFactorization.hpp
 *
 *  Created on: Jul 15, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_ORTHOGONALFACTORIZATION_HPP_
#define DOTK_ORTHOGONALFACTORIZATION_HPP_

#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename Type>
class matrix;

class DOTk_OrthogonalFactorization
{
public:
    DOTk_OrthogonalFactorization(dotk::types::qr_t type_);
    virtual ~DOTk_OrthogonalFactorization();

    dotk::types::qr_t type() const;
    virtual void factorization(const std::tr1::shared_ptr<dotk::matrix<Real> > & input_,
                               std::tr1::shared_ptr<dotk::matrix<Real> > & Q_,
                               std::tr1::shared_ptr<dotk::matrix<Real> > & R_) = 0;

private:
    dotk::types::qr_t m_OrthogonalFactorizationType;

private:
    DOTk_OrthogonalFactorization(const dotk::DOTk_OrthogonalFactorization &);
    dotk::DOTk_OrthogonalFactorization & operator=(const dotk::DOTk_OrthogonalFactorization & rhs_);
};

}

#endif /* DOTK_ORTHOGONALFACTORIZATION_HPP_ */
