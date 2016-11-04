/*
 * DOTk_DoubleDoglegTrustRegion.hpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_DOUBLEDOGLEGTRUSTREGION_HPP_
#define DOTK_DOUBLEDOGLEGTRUSTREGION_HPP_

#include "DOTk_TrustRegion.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_DoubleDoglegTrustRegion : public dotk::DOTk_TrustRegion
{
public:
    explicit DOTk_DoubleDoglegTrustRegion(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_);
    virtual ~DOTk_DoubleDoglegTrustRegion();

    Real getParamPromotesMonotonicallyDecreasingQuadraticModel() const;
    void setParamPromotesMonotonicallyDecreasingQuadraticModel(Real value_);

    Real computeDoubleDoglegRoot(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & newton_direction_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_grad_);

    void doubleDogleg(const Real & trust_region_radius_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & newton_step_);

    virtual void step(const dotk::DOTk_OptimizationDataMng * const mng_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_grad_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & scaled_direction_);

private:
    void computeScaledNewtonStep(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_grad_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & newton_step_);
    void computeConvexCombinationBetweenCauchyAndDoglegStep(const Real & trust_region_radius_,
                                                            const std::tr1::shared_ptr<dotk::vector<Real> > & newton_step_);

private:
    Real mParamPromoteMonotonicallyDecreasingQuadraticModel;

    std::tr1::shared_ptr<dotk::vector<Real> > mCauchyPoint;
    std::tr1::shared_ptr<dotk::vector<Real> > mScaledNewtonStep;

private:
    DOTk_DoubleDoglegTrustRegion(const dotk::DOTk_DoubleDoglegTrustRegion &);
    dotk::DOTk_DoubleDoglegTrustRegion operator=(const dotk::DOTk_DoubleDoglegTrustRegion &);
};

}

#endif /* DOTK_DOUBLEDOGLEGTRUSTREGION_HPP_ */
