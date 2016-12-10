/*
 * TRROM_Dual.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Types.hpp"
#include "TRROM_Dual.hpp"

namespace trrom
{

Dual::Dual() :
        trrom::Variable(trrom::types::DUAL)
{
}

Dual::Dual(const trrom::Vector<double> & data_) :
        trrom::Variable(trrom::types::DUAL, data_)
{
}

Dual::Dual(const trrom::Vector<double> & data_,
           const trrom::Vector<double> & lower_bound_,
           const trrom::Vector<double> & upper_bound_) :
        trrom::Variable(trrom::types::DUAL, data_, lower_bound_, upper_bound_)
{
}

Dual::~Dual()
{
}

}
