/*
 * DOTk_LinearOperator.hpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LINEAROPERATOR_HPP_
#define DOTK_LINEAROPERATOR_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_LinearOperator
{
public:
    explicit DOTk_LinearOperator(dotk::types::linear_operator_t type_);
    virtual ~DOTk_LinearOperator();

    dotk::types::linear_operator_t type() const;

    virtual void setNumOtimizationItrDone(size_t itr_);
    virtual void updateLimitedMemoryStorage(bool update_);
    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & vector_,
                       const std::tr1::shared_ptr<dotk::Vector<Real> > & output_);

private:
    dotk::types::linear_operator_t m_LinearOperatorType;

private:
    DOTk_LinearOperator(const dotk::DOTk_LinearOperator &);
    dotk::DOTk_LinearOperator & operator=(const dotk::DOTk_LinearOperator &);
};

}

#endif /* DOTK_LINEAROPERATOR_HPP_ */
