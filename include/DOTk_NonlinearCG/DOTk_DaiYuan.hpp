/*
 * DOTk_DaiYuan.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DAIYUAN_HPP_
#define DOTK_DAIYUAN_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename ScalarType>
class Vector;

class DOTk_DaiYuan : public dotk::DOTk_DescentDirection
{
public:
    DOTk_DaiYuan();
    virtual ~DOTk_DaiYuan();

    Real computeScaleFactor(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                            const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                            const std::shared_ptr<dotk::Vector<Real> > & dir_);
    void getDirection(const std::shared_ptr<dotk::Vector<Real> > & old_grad_,
                      const std::shared_ptr<dotk::Vector<Real> > & new_grad_,
                      const std::shared_ptr<dotk::Vector<Real> > & dir_);
    virtual void direction(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    DOTk_DaiYuan(const dotk::DOTk_DaiYuan &);
    dotk::DOTk_DaiYuan & operator=(const dotk::DOTk_DaiYuan &);
};

}

#endif /* DOTK_DAIYUAN_HPP_ */
