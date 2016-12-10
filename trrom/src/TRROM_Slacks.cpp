/*
 * TRROM_Slacks.cpp
 *
 *  Created on: Sep 3, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Vector.hpp"
#include "TRROM_Slacks.hpp"

namespace trrom
{

Slacks::Slacks() :
        trrom::Variable(trrom::types::SLACKS)
{
}

Slacks::Slacks(const trrom::Vector<double> & data_) :
        trrom::Variable(trrom::types::SLACKS, data_)
{
}

Slacks::Slacks(const trrom::Vector<double> & data_,
               const trrom::Vector<double> & lower_bound_,
               const trrom::Vector<double> & upper_bound_) :
        trrom::Variable(trrom::types::SLACKS, data_, lower_bound_, upper_bound_)
{
}

Slacks::~Slacks()
{
}

}
