/*
 * DOTk_NumericalDifferentiatonFactory.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_
#define DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_

#include <memory>
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
    explicit DOTk_NumericalDifferentiatonFactory(dotk::types::numerical_integration_t aType);
    ~DOTk_NumericalDifferentiatonFactory();

    dotk::types::numerical_integration_t type() const;
    void type(dotk::types::numerical_integration_t aType);

    void buildForwardDifferenceHessian(const dotk::Vector<Real> & aInput,
                                       std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput);
    void buildBackwardDifferenceHessian(const dotk::Vector<Real> & aInput,
                                        std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput);
    void buildCentralDifferenceHessian(const dotk::Vector<Real> & aInput,
                                       std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput);
    void buildSecondOrderForwardDifferenceHessian(const dotk::Vector<Real> & aInput,
                                                  std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput);
    void buildThirdOrderForwardDifferenceHessian(const dotk::Vector<Real> & aInput,
                                                 std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput);
    void buildThirdOrderBackwardDifferenceHessian(const dotk::Vector<Real> & aInput,
                                                  std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput);
    void build(const dotk::Vector<Real> & aInput, std::shared_ptr<dotk::DOTk_NumericalDifferentiation> & aOutput);

private:
    dotk::types::numerical_integration_t m_Type;

private:
    DOTk_NumericalDifferentiatonFactory(const dotk::DOTk_NumericalDifferentiatonFactory &);
    dotk::DOTk_NumericalDifferentiatonFactory operator=(const dotk::DOTk_NumericalDifferentiatonFactory &);
};

}

#endif /* DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_ */
