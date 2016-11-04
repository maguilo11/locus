/*
 * DOTk_NumericalDifferentiatonFactory.hpp
 *
 *  Created on: Jan 31, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_
#define DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_

namespace dotk
{

class DOTk_Primal;
class DOTk_NumericalDifferentiation;

class DOTk_NumericalDifferentiatonFactory
{
public:
    DOTk_NumericalDifferentiatonFactory();
    DOTk_NumericalDifferentiatonFactory(dotk::types::numerical_integration_t type_);
    ~DOTk_NumericalDifferentiatonFactory();

    dotk::types::numerical_integration_t type() const;
    void type(dotk::types::numerical_integration_t type_);

    void buildForwardDifferenceHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                       std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & intg_);
    void buildBackwardDifferenceHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                        std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & intg_);
    void buildCentralDifferenceHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                       std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & intg_);
    void buildSecondOrderForwardDifferenceHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                  std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & intg_);
    void buildThirdOrderForwardDifferenceHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                 std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & intg_);
    void buildThirdOrderBackwardDifferenceHessian(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                                                  std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & intg_);
    void build(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
               std::tr1::shared_ptr<dotk::DOTk_NumericalDifferentiation> & intg_);

private:
    dotk::types::numerical_integration_t m_Type;

private:
    DOTk_NumericalDifferentiatonFactory(const dotk::DOTk_NumericalDifferentiatonFactory &);
    dotk::DOTk_NumericalDifferentiatonFactory operator=(const dotk::DOTk_NumericalDifferentiatonFactory &);
};

}

#endif /* DOTK_NUMERICALDIFFERENTIATIONFACTORY_HPP_ */
