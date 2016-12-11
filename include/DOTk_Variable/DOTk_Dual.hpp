/*
 * DOTk_Dual.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DUAL_HPP_
#define DOTK_DUAL_HPP_

#include "DOTk_Variable.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_Dual: public dotk::DOTk_Variable
{
public:
    DOTk_Dual();
    explicit DOTk_Dual(const dotk::Vector<Real> & data_);
    DOTk_Dual(const dotk::Vector<Real> & data_,
              const dotk::Vector<Real> & lower_bound_,
              const dotk::Vector<Real> & upper_bound_);
    virtual ~DOTk_Dual();

    size_t getDualBasisSize() const;
    void setDualBasisSize(size_t size_);

private:
    size_t m_DualBasisSize;

private:
    DOTk_Dual(const dotk::DOTk_Dual &);
    dotk::DOTk_Dual & operator=(const dotk::DOTk_Dual &);
};

}

#endif /* DOTK_DUAL_HPP_ */
