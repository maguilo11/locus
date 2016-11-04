/*
 * DOTk_AugmentedSystem.hpp
 *
 *  Created on: Feb 23, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_AUGMENTEDSYSTEM_HPP_
#define DOTK_AUGMENTEDSYSTEM_HPP_

#include "DOTk_LinearOperator.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_AugmentedSystem: public dotk::DOTk_LinearOperator
{
public:
    DOTk_AugmentedSystem();
    virtual ~DOTk_AugmentedSystem();

    virtual void apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & data_,
                       const std::tr1::shared_ptr<dotk::vector<Real> > & output_);

private:
    DOTk_AugmentedSystem(const dotk::DOTk_AugmentedSystem &);
    dotk::DOTk_AugmentedSystem & operator=(const dotk::DOTk_AugmentedSystem &);
};

}

#endif /* DOTK_AUGMENTEDSYSTEM_HPP_ */
