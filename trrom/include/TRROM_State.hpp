/*
 * TRROM_State.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_STATE_HPP_
#define TRROM_STATE_HPP_

#include "TRROM_Variable.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class State : public trrom::Variable
{
public:
    State();
    explicit State(const trrom::Vector<double> & data_);
    State(const trrom::Vector<double> & data_,
          const trrom::Vector<double> & lower_bound_,
          const trrom::Vector<double> & upper_bound_);
    virtual ~State();

private:
    State(const trrom::State &);
    trrom::State & operator=(const trrom::State &);
};

}

#endif /* TRROM_STATE_HPP_ */
