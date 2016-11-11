/*
 * TRROM_Slacks.hpp
 *
 *  Created on: Sep 3, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_SLACKS_HPP_
#define TRROM_SLACKS_HPP_

#include "TRROM_Variable.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Slacks : public trrom::Variable
{
public:
    Slacks();
    explicit Slacks(const trrom::Vector<double> & data_);
    Slacks(const trrom::Vector<double> & data_,
           const trrom::Vector<double> & lower_bound_,
           const trrom::Vector<double> & upper_bound_);
    virtual ~Slacks();

private:
    Slacks(const trrom::Slacks &);
    trrom::Slacks & operator=(const trrom::Slacks &);
};

}

#endif /* TRROM_SLACKS_HPP_ */
