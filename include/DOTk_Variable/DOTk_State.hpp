/*
 * DOTk_State.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_STATE_HPP_
#define DOTK_STATE_HPP_

#include "DOTk_Variable.hpp"

namespace dotk
{

class DOTk_State: public dotk::DOTk_Variable
{
public:
    DOTk_State();
    explicit DOTk_State(const dotk::vector<Real> & data_);
    DOTk_State(const dotk::vector<Real> & data_,
               const dotk::vector<Real> & lower_bound_,
               const dotk::vector<Real> & upper_bound_);
    virtual ~DOTk_State();

    size_t getStateBasisSize() const;
    void setStateBasisSize(size_t size_);

private:
    size_t m_StateBasisSize;

private:
    DOTk_State(const dotk::DOTk_State &);
    dotk::DOTk_State & operator=(const dotk::DOTk_State &);
};

}

#endif /* DOTK_STATE_HPP_ */
