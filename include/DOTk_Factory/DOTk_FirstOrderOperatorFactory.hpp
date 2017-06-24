/*
 * DOTk_FirstOrderOperatorFactory.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_FIRSTORDEROPERATORFACTORY_HPP_
#define DOTK_FIRSTORDEROPERATORFACTORY_HPP_

#include <memory>
#include <DOTk_Types.hpp>

namespace dotk
{

class DOTk_FirstOrderOperator;
class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_FirstOrderOperatorFactory
{
public:
    DOTk_FirstOrderOperatorFactory();
    explicit DOTk_FirstOrderOperatorFactory(dotk::types::gradient_t aType);
    ~DOTk_FirstOrderOperatorFactory();

    void setFactoryType(dotk::types::gradient_t aType);
    dotk::types::gradient_t getFactoryType() const;

    void buildForwardFiniteDiffGradient(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                        std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
    void buildBackwardFiniteDiffGradient(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                         std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
    void buildCentralFiniteDiffGradient(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                        std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
    void buildUserDefinedGradient(std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
    void buildParallelForwardFiniteDiffGradient(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                                std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
    void buildParallelBackwardFiniteDiffGradient(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                                 std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
    void buildParallelCentralFiniteDiffGradient(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                                std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
    void build(const dotk::DOTk_OptimizationDataMng * const aMng,
               std::shared_ptr<dotk::DOTk_FirstOrderOperator> & aOutput);
private:
    dotk::types::gradient_t mFactoryType;

private:
    DOTk_FirstOrderOperatorFactory(const dotk::DOTk_FirstOrderOperatorFactory &);
    DOTk_FirstOrderOperatorFactory operator=(const dotk::DOTk_FirstOrderOperatorFactory &);
};

}

#endif /* DOTK_FIRSTORDEROPERATORFACTORY_HPP_ */
