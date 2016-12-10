/*
 * TRROM_Control.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_CONTROL_HPP_
#define TRROM_CONTROL_HPP_

#include "TRROM_Variable.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Control : public trrom::Variable
{
public:
    Control();
    explicit Control(const trrom::Vector<double> & data_);
    Control(const trrom::Vector<double> & data_,
            const trrom::Vector<double> & lower_bound_,
            const trrom::Vector<double> & upper_bound_);
    virtual ~Control();

private:
    Control(const trrom::Control &);
    trrom::Control & operator=(const trrom::Control &);
};

}

#endif /* TRROM_CONTROL_HPP_ */
