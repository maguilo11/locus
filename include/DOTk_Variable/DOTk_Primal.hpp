/*
 * DOTk_Primal.hpp
 *
 *  Created on: Jun 14, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PRIMAL_HPP_
#define DOTK_PRIMAL_HPP_

#include <tr1/memory>

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_Dual;
class DOTk_State;
class DOTk_Control;

template<typename ScalarType>
class Vector;

class DOTk_Primal
{
public:
    DOTk_Primal();
    ~DOTk_Primal();

    const std::tr1::shared_ptr<dotk::Vector<Real> > & dual() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & state() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & control() const;

    dotk::types::variable_t type() const;

    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDualLowerBound() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getDualUpperBound() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getStateLowerBound() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getStateUpperBound() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getControlLowerBound() const;
    const std::tr1::shared_ptr<dotk::Vector<Real> > & getControlUpperBound() const;

    void setDualBasisSize(size_t size_);
    void setDualLowerBound(Real value_);
    void setDualUpperBound(Real value_);
    void setDualLowerBound(const dotk::Vector<Real> & lower_bound_);
    void setDualUpperBound(const dotk::Vector<Real> & upper_bound_);

    void setStateBasisSize(size_t size_);
    void setStateLowerBound(Real value_);
    void setStateUpperBound(Real value_);
    void setStateLowerBound(const dotk::Vector<Real> & lower_bound_);
    void setStateUpperBound(const dotk::Vector<Real> & upper_bound_);

    void setControlBasisSize(size_t size_);
    void setControlLowerBound(Real value_);
    void setControlUpperBound(Real value_);
    void setControlLowerBound(const dotk::Vector<Real> & lower_bound_);
    void setControlUpperBound(const dotk::Vector<Real> & upper_bound_);

    void allocateUserDefinedDual(const dotk::Vector<Real> & dual_);
    void allocateSerialDualArray(size_t size_, Real value_ = 0.);
    void allocateSerialDualVector(size_t size_, Real value_ = 0.);

    void allocateUserDefinedState(const dotk::Vector<Real> & state_);
    void allocateSerialStateArray(size_t size_, Real value_ = 0.);
    void allocateSerialStateVector(size_t size_, Real value_ = 0.);

    void allocateUserDefinedControl(const dotk::Vector<Real> & control_);
    void allocateSerialControlArray(size_t size_, Real value_ = 0.);
    void allocateSerialControlVector(size_t size_, Real value_ = 0.);

private:
    std::tr1::shared_ptr<dotk::DOTk_Dual> m_Dual;
    std::tr1::shared_ptr<dotk::DOTk_State> m_State;
    std::tr1::shared_ptr<dotk::DOTk_Control> m_Control;

private:
    DOTk_Primal(const dotk::DOTk_Primal &);
    dotk::DOTk_Primal & operator=(const dotk::DOTk_Primal & rhs_);
};

}

#endif /* DOTK_PRIMAL_HPP_ */
