/*
 * DOTk_DescentDirection.hpp
 *
 *  Created on: Sep 11, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_DESCENTDIRECTION_HPP_
#define DOTK_DESCENTDIRECTION_HPP_

#include "DOTk_Types.hpp"

namespace dotk
{

class DOTk_OptimizationDataMng;

template<class Type>
class vector;

class DOTk_DescentDirection
{
public:
    DOTk_DescentDirection(dotk::types::nonlinearcg_t dir_);
    virtual ~DOTk_DescentDirection();

    void setScaleFactor(Real factor_);
    Real getScaleFactor() const;
    void setMinCosineAngleTol(Real tol_);
    Real getMinCosineAngleTol() const;

    void setNonlinearCGType(dotk::types::nonlinearcg_t type_);
    dotk::types::nonlinearcg_t getNonlinearCGType() const;

    Real computeCosineAngle(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                            const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);
    bool isTrialStepOrthogonalToSteepestDescent(Real cosine_val_);
    void steepestDescent(const std::tr1::shared_ptr<dotk::vector<Real> > & grad_,
                         const std::tr1::shared_ptr<dotk::vector<Real> > & dir_);

    virtual void direction(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_) = 0;

private:
    Real mScaleFactor;
    Real mMinCosineAngleTol;
    dotk::types::nonlinearcg_t mNonlinearCGType;

private:
    DOTk_DescentDirection(const dotk::DOTk_DescentDirection &);
    dotk::DOTk_DescentDirection & operator=(const dotk::DOTk_DescentDirection & rhs_)
    {
        // check for self-assignment by comparing the address of the
        // implicit object and the parameter
        if(this == &rhs_)
        {
            return (*this);
        }
        // return the existing object
        return (*this);
    }

};

}

#endif /* DOTK_DESCENTDIRECTION_HPP_ */
