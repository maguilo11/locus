/*
 * DOTk_NumericalDifferentiatonFactory.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_
#define DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_

#include <tr1/memory>
#include "DOTk_Types.hpp"

namespace dotk
{

template<typename ScalarType>
class Vector;

class DOTk_NumericalDifferentiation;

class DOTk_NumericalDifferentiatonFactory
{
public:
    DOTk_NumericalDifferentiatonFactory();
    explicit DOTk_NumericalDifferentiatonFactory(dotk::types::numerical_integration_t type_);
    ~DOTk_NumericalDifferentiatonFactory();

    dotk::types::numerical_integration_t type() const;
    void type(dotk::types::numerical_integration_t type_);

    void buildForwardDifferenceHessian(const dotk::Vector<Real> & input_,
                                       std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_);
    void buildBackwardDifferenceHessian(const dotk::Vector<Real> & input_,
                                        std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_);
    void buildCentralDifferenceHessian(const dotk::Vector<Real> & input_,
                                       std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_);
    void buildSecondOrderForwardDifferenceHessian(const dotk::Vector<Real> & input_,
                                                  std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_);
    void buildThirdOrderForwardDifferenceHessian(const dotk::Vector<Real> & input_,
                                                 std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_);
    void buildThirdOrderBackwardDifferenceHessian(const dotk::Vector<Real> & input_,
                                                  std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_);
    void build(const dotk::Vector<Real> & input_, std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & output_);

private:
    dotk::types::numerical_integration_t m_Type;

private:
    DOTk_NumericalDifferentiatonFactory(const dotk::DOTk_NumericalDifferentiatonFactory &);
    dotk::DOTk_NumericalDifferentiatonFactory operator=(const dotk::DOTk_NumericalDifferentiatonFactory &);
};

}

#endif /* DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_ */
