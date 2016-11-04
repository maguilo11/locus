/*
 * DOTk_HagerZhang.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_HAGERZHANG_HPP_
#define DOTK_HAGERZHANG_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_HagerZhang : public dotk::DOTk_DescentDirection
{
public:
    DOTk_HagerZhang();
    virtual ~DOTk_HagerZhang();

    void setLowerBoundLimit(Real value_);
    Real getLowerBoundLimit() const;
    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real mLowerBoundLimit;

private:
    DOTk_HagerZhang(const dotk::DOTk_HagerZhang &);
    dotk::DOTk_HagerZhang & operator=(const dotk::DOTk_HagerZhang &);
};

}

#endif /* DOTK_HAGERZHANG_HPP_ */
