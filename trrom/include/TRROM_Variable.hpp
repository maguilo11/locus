/*
 * TRROM_Variable.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_VARIABLE_HPP_
#define TRROM_VARIABLE_HPP_

#include "TRROM_Types.hpp"

namespace trrom
{

template<typename ScalarType>
class Vector;

class Variable
{
public:
    explicit Variable(trrom::types::variable_t type_);
    Variable(trrom::types::variable_t type_, const trrom::Vector<double> & data_);
    Variable(trrom::types::variable_t type_,
             const trrom::Vector<double> & data_,
             const trrom::Vector<double> & lower_bound_,
             const trrom::Vector<double> & upper_bound_);
    ~Variable();

    int size() const;
    trrom::types::variable_t type() const;
    const std::tr1::shared_ptr<trrom::Vector<double> > & data() const;

    void setLowerBound(double value_);
    void setLowerBound(const trrom::Vector<double> & lower_bound_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & lowerBound() const;

    void setUpperBound(double value_);
    void setUpperBound(const trrom::Vector<double> & upper_bound_);
    const std::tr1::shared_ptr<trrom::Vector<double> > & upperBound() const;

private:
    void checkData();
    void initialize(const trrom::Vector<double> & data_);
    void initialize(const trrom::Vector<double> & data_,
                    const trrom::Vector<double> & lower_bound_,
                    const trrom::Vector<double> & upper_bound_);

private:
    trrom::types::variable_t m_Type;
    std::tr1::shared_ptr<trrom::Vector<double> > m_Data;
    std::tr1::shared_ptr<trrom::Vector<double> > m_LowerBound;
    std::tr1::shared_ptr<trrom::Vector<double> > m_UpperBound;

private:
    Variable(const trrom::Variable &);
    trrom::Variable & operator=(const trrom::Variable &);
};

}

#endif /* TRROM_VARIABLE_HPP_ */
