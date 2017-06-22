/*
 * DOTk_Variable.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_VARIABLE_HPP_
#define DOTK_VARIABLE_HPP_

#include <memory>

#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_Variable
{
public:
    explicit DOTk_Variable(dotk::types::variable_t type_);
    DOTk_Variable(dotk::types::variable_t type_, const dotk::Vector<Real> & data_);
    DOTk_Variable(dotk::types::variable_t type_,
                  const dotk::Vector<Real> & data_,
                  const dotk::Vector<Real> & lower_bound_,
                  const dotk::Vector<Real> & upper_bound_);
    ~DOTk_Variable();

    size_t size() const;
    dotk::types::variable_t type() const;
    const std::shared_ptr<dotk::Vector<Real> > & data() const;

    void setLowerBound(Real value_);
    void setLowerBound(const dotk::Vector<Real> & lower_bound_);
    const std::shared_ptr<dotk::Vector<Real> > & lowerBound() const;

    void setUpperBound(Real value_);
    void setUpperBound(const dotk::Vector<Real> & upper_bound_);
    const std::shared_ptr<dotk::Vector<Real> > & upperBound() const;

    void allocate(const dotk::Vector<Real> & input_);
    void allocateSerialArray(size_t size_, Real value_);
    void allocateSerialVector(size_t size_, Real value_);

private:
    void checkData();
    void initialize(const dotk::Vector<Real> & data_);
    void initialize(const dotk::Vector<Real> & data_,
                    const dotk::Vector<Real> & lower_bound_,
                    const dotk::Vector<Real> & upper_bound_);

private:
    dotk::types::variable_t m_Type;
    std::shared_ptr<dotk::Vector<Real> > m_Data;
    std::shared_ptr<dotk::Vector<Real> > m_LowerBound;
    std::shared_ptr<dotk::Vector<Real> > m_UpperBound;

private:
    DOTk_Variable(const dotk::DOTk_Variable &);
    dotk::DOTk_Variable & operator=(const dotk::DOTk_Variable &);
};

}

#endif /* DOTK_VARIABLE_HPP_ */
