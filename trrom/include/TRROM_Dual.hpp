/*
 * TRROM_Dual.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_DUAL_HPP_
#define TRROM_DUAL_HPP_

#include "TRROM_Variable.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Dual : public trrom::Variable
{
public:
    Dual();
    explicit Dual(const trrom::Vector<double> & data_);
    Dual(const trrom::Vector<double> & data_,
         const trrom::Vector<double> & lower_bound_,
         const trrom::Vector<double> & upper_bound_);
    virtual ~Dual();

private:
    Dual(const trrom::Dual &);
    trrom::Dual & operator=(const trrom::Dual &);
};

}

#endif /* TRROM_DUAL_HPP_ */
