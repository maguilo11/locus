/*
 * DOTk_DaiYuanHybrid.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DAIYUANHYBRID_HPP_
#define DOTK_DAIYUANHYBRID_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_DaiYuanHybrid : public dotk::DOTk_DescentDirection
{
public:
    DOTk_DaiYuanHybrid();
    virtual ~DOTk_DaiYuanHybrid();

    Real getWolfeConstant() const;
    void setWolfeConstant(Real value_);
    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::Vector<Real> > & old_grad_,
    const std::tr1::shared_ptr<dotk::Vector<Real> > & new_grad_,
    const std::tr1::shared_ptr<dotk::Vector<Real> > & dir_);
    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real mWolfeConstant;

private:
    DOTk_DaiYuanHybrid(const dotk::DOTk_DaiYuanHybrid &);
    dotk::DOTk_DaiYuanHybrid & operator=(const dotk::DOTk_DaiYuanHybrid &);
};

}

#endif /* DOTK_DAIYUANHYBRID_HPP_ */
