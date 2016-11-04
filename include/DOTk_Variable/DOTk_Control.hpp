/*
 * DOTk_Control.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_CONTROL_HPP_
#define DOTK_CONTROL_HPP_

#include "DOTk_Variable.hpp"

namespace dotk
{

class DOTk_Control: public dotk::DOTk_Variable
{
public:
    DOTk_Control();
    explicit DOTk_Control(const dotk::vector<Real> & data_);
    DOTk_Control(const dotk::vector<Real> & data_,
                 const dotk::vector<Real> & lower_bound_,
                 const dotk::vector<Real> & upper_bound_);
    virtual ~DOTk_Control();

    size_t getControlBasisSize() const;
    void setControlBasisSize(size_t size_);

private:
    size_t m_ControlBasisSize;

private:
    DOTk_Control(const dotk::DOTk_Control &);
    dotk::DOTk_Control & operator=(const dotk::DOTk_Control &);
};

}

#endif /* DOTK_CONTROL_HPP_ */
