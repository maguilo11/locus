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

template <typename Type>
class vector;

class DOTk_Primal
{
public:
    DOTk_Primal();
    ~DOTk_Primal();

    const std::tr1::shared_ptr<dotk::vector<Real> > & dual() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & state() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & control() const;

    const std::tr1::shared_ptr<dotk::DOTk_Dual> & dualStruc() const;
    const std::tr1::shared_ptr<dotk::DOTk_State> & stateStruc() const;
    const std::tr1::shared_ptr<dotk::DOTk_Control> & controlStruc() const;

    size_t getDualBasisSize() const;
    size_t getStateBasisSize() const;
    size_t getControlBasisSize() const;
    dotk::types::variable_t type() const;

    const std::tr1::shared_ptr<dotk::vector<Real> > & getDualLowerBound() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getDualUpperBound() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getStateLowerBound() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getStateUpperBound() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getControlLowerBound() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & getControlUpperBound() const;

    void setDualBasisSize(size_t size_);
    void setDualLowerBound(Real value_);
    void setDualUpperBound(Real value_);
    void setDualLowerBound(const dotk::vector<Real> & lower_bound_);
    void setDualUpperBound(const dotk::vector<Real> & upper_bound_);

    void setStateBasisSize(size_t size_);
    void setStateLowerBound(Real value_);
    void setStateUpperBound(Real value_);
    void setStateLowerBound(const dotk::vector<Real> & lower_bound_);
    void setStateUpperBound(const dotk::vector<Real> & upper_bound_);

    void setControlBasisSize(size_t size_);
    void setControlLowerBound(Real value_);
    void setControlUpperBound(Real value_);
    void setControlLowerBound(const dotk::vector<Real> & lower_bound_);
    void setControlUpperBound(const dotk::vector<Real> & upper_bound_);

    void allocateUserDefinedDual(const dotk::vector<Real> & dual_);
    void allocateSerialDualArray(size_t size_, Real value_ = 0.);
    void allocateSerialDualVector(size_t size_, Real value_ = 0.);

    void allocateUserDefinedState(const dotk::vector<Real> & state_);
    void allocateSerialStateArray(size_t size_, Real value_ = 0.);
    void allocateSerialStateVector(size_t size_, Real value_ = 0.);

    void allocateUserDefinedControl(const dotk::vector<Real> & control_);
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
