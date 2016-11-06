/*
 * DOTk_DoglegTrustRegion.hpp
 *
 *  Created on: Sep 10, 2014
 *      Author: Miguel A. Aguilo Valentin (maguilo@sandia.gov)
 */

#ifndef DOTK_DOGLEGTRUSTREGION_HPP_
#define DOTK_DOGLEGTRUSTREGION_HPP_

#include "DOTk_TrustRegion.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<typename Type>
class vector;

class DOTk_DoglegTrustRegion : public dotk::DOTk_TrustRegion
{
public:
    DOTk_DoglegTrustRegion();
    virtual ~DOTk_DoglegTrustRegion();

    void dogleg(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_grad_,
                const std::tr1::shared_ptr<dotk::vector<Real> > & cauchy_dir_,
                const std::tr1::shared_ptr<dotk::vector<Real> > & newton_dir_);

    virtual void step(const dotk::DOTk_OptimizationDataMng * const mng_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & cauchy_direction_,
                      const std::tr1::shared_ptr<dotk::vector<Real> > & scaled_direction_);

private:
    DOTk_DoglegTrustRegion(const dotk::DOTk_DoglegTrustRegion &);
    DOTk_DoglegTrustRegion operator=(const dotk::DOTk_DoglegTrustRegion &);
};

}

#endif /* DOTK_DOGLEGTRUSTREGION_HPP_ */
