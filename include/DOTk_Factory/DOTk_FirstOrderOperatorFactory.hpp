/*
 * DOTk_FirstOrderOperatorFactory.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_FIRSTORDEROPERATORFACTORY_HPP_
#define DOTK_FIRSTORDEROPERATORFACTORY_HPP_

#include <tr1/memory>
#include <DOTk_Types.hpp>

namespace dotk
{

class DOTk_AssemblyManager;
class DOTk_FirstOrderOperator;
class DOTk_OptimizationDataMng;

class DOTk_FirstOrderOperatorFactory
{
public:
    DOTk_FirstOrderOperatorFactory();
    explicit DOTk_FirstOrderOperatorFactory(dotk::types::gradient_t type_);
    ~DOTk_FirstOrderOperatorFactory();

    void setFactoryType(dotk::types::gradient_t type_);
    dotk::types::gradient_t getFactoryType() const;

    void buildForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                        std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
    void buildBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                         std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
    void buildCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                        std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
    void buildUserDefinedGradient(std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
    void buildParallelForwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
    void buildParallelBackwardFiniteDiffGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                 std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
    void buildParallelCentralFiniteDiffGradient(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                                std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
    void build(const dotk::DOTk_OptimizationDataMng * const mng_,
               std::tr1::shared_ptr<dotk::DOTk_FirstOrderOperator> & grad_);
private:
    dotk::types::gradient_t mFactoryType;

private:
    DOTk_FirstOrderOperatorFactory(const dotk::DOTk_FirstOrderOperatorFactory &);
    DOTk_FirstOrderOperatorFactory operator=(const dotk::DOTk_FirstOrderOperatorFactory &);
};

}

#endif /* DOTK_FIRSTORDEROPERATORFACTORY_HPP_ */
