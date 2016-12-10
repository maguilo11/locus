/*
 * TRROM_State.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Types.hpp"
#include "TRROM_State.hpp"

namespace trrom
{

State::State() :
        trrom::Variable(trrom::types::STATE)
{
}

State::State(const trrom::Vector<double> & data_) :
        trrom::Variable(trrom::types::STATE, data_)
{
}

State::State(const trrom::Vector<double> & data_,
             const trrom::Vector<double> & lower_bound_,
             const trrom::Vector<double> & upper_bound_) :
        trrom::Variable(trrom::types::STATE, data_, lower_bound_, upper_bound_)
{
}

State::~State()
{
}

}
