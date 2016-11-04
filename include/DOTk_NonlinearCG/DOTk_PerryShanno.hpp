/*
 * DOTk_PerryShanno.hpp
 *
 *  Created on: Sep 12, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PERRYSHANNO_HPP_
#define DOTK_PERRYSHANNO_HPP_

#include "DOTk_DescentDirection.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_PerryShanno : public dotk::DOTk_DescentDirection
{
public:
    DOTk_PerryShanno();
    virtual ~DOTk_PerryShanno();

    void setAlphaScaleFactor(Real value_);
    Real getAlphaScaleFactor() const;
    void setThetaScaleFactor(Real value_);
    Real getThetaScaleFactor() const;
    void setLowerBoundLimit(Real value_);
    Real getLowerBoundLimit() const;
    Real computeAlphaScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    Real computeThetaScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_);
    Real computeScaleFactor(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    void getDirection(const std::tr1::shared_ptr<dotk::vector<Real> > & old_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & old_primal_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & new_primal_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_);

private:
    Real mAlpha;
    Real mTheta;
    Real mLowerBoundLimit;

private:
    DOTk_PerryShanno(const dotk::DOTk_PerryShanno &);
    dotk::DOTk_PerryShanno & operator=(const dotk::DOTk_PerryShanno &);
};

}

#endif /* DOTK_PERRYSHANNO_HPP_ */
