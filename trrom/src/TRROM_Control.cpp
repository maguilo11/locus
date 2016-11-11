/*
 * TRROM_Control.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include "TRROM_Types.hpp"
#include "TRROM_Control.hpp"

namespace trrom
{

Control::Control() :
        trrom::Variable(trrom::types::CONTROL)
{
}

Control::Control(const trrom::Vector<double> & data_) :
        trrom::Variable(trrom::types::CONTROL, data_)
{
}

Control::Control(const trrom::Vector<double> & data_,
                 const trrom::Vector<double> & lower_bound_,
                 const trrom::Vector<double> & upper_bound_) :
        trrom::Variable(trrom::types::CONTROL, data_, lower_bound_, upper_bound_)
{
}

Control::~Control()
{
}

}
