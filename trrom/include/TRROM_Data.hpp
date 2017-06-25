/*
 * TRROM_Data.hpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DATA_HPP_
#define DATA_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

class Dual;
class State;
class Slacks;
class Control;

template <typename ScalarType>
class Vector;

class Data
{
public:
    Data();
    ~Data();

    const std::shared_ptr<trrom::Vector<double> > & dual() const;
    const std::shared_ptr<trrom::Vector<double> > & state() const;
    const std::shared_ptr<trrom::Vector<double> > & slacks() const;
    const std::shared_ptr<trrom::Vector<double> > & control() const;

    const std::shared_ptr<trrom::Vector<double> > & getDualLowerBound() const;
    const std::shared_ptr<trrom::Vector<double> > & getDualUpperBound() const;
    const std::shared_ptr<trrom::Vector<double> > & getStateLowerBound() const;
    const std::shared_ptr<trrom::Vector<double> > & getStateUpperBound() const;
    const std::shared_ptr<trrom::Vector<double> > & getSlacksLowerBound() const;
    const std::shared_ptr<trrom::Vector<double> > & getSlacksUpperBound() const;
    const std::shared_ptr<trrom::Vector<double> > & getControlLowerBound() const;
    const std::shared_ptr<trrom::Vector<double> > & getControlUpperBound() const;

    void setDualLowerBound(double value_);
    void setDualUpperBound(double value_);
    void setDualLowerBound(const trrom::Vector<double> & lower_bound_);
    void setDualUpperBound(const trrom::Vector<double> & upper_bound_);

    void setStateLowerBound(double value_);
    void setStateUpperBound(double value_);
    void setStateLowerBound(const trrom::Vector<double> & lower_bound_);
    void setStateUpperBound(const trrom::Vector<double> & upper_bound_);

    void setSlacksLowerBound(double value_);
    void setSlacksUpperBound(double value_);
    void setSlacksLowerBound(const trrom::Vector<double> & lower_bound_);
    void setSlacksUpperBound(const trrom::Vector<double> & upper_bound_);

    void setControlLowerBound(double value_);
    void setControlUpperBound(double value_);
    void setControlLowerBound(const trrom::Vector<double> & lower_bound_);
    void setControlUpperBound(const trrom::Vector<double> & upper_bound_);

    void allocateDual(const trrom::Vector<double> & dual_);
    void allocateState(const trrom::Vector<double> & state_);
    void allocateSlacks(const trrom::Vector<double> & slacks_);
    void allocateControl(const trrom::Vector<double> & control_);

private:
    std::shared_ptr<trrom::Dual> m_Dual;
    std::shared_ptr<trrom::State> m_State;
    std::shared_ptr<trrom::Slacks> m_Slacks;
    std::shared_ptr<trrom::Control> m_Control;

private:
    Data(const trrom::Data &);
    trrom::Data & operator=(const trrom::Data & rhs_);
};

}

#endif /* DATA_HPP_ */
