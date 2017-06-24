/*
 * DOTk_PrecGenMinResDataMng.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_PRECGENMINRESDATAMNG_HPP_
#define DOTK_PRECGENMINRESDATAMNG_HPP_

#include "DOTk_KrylovSolverDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;
class DOTk_RightPreconditioner;
class DOTk_OrthogonalProjection;

template<typename ScalarType>
class Vector;

class DOTk_PrecGenMinResDataMng : public dotk::DOTk_KrylovSolverDataMng
{
public:
    DOTk_PrecGenMinResDataMng(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                              const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator,
                              size_t aMaxNumIterations  = 200);
    virtual ~DOTk_PrecGenMinResDataMng();

    void setDirectSolver(dotk::types::direct_solver_t type_);
    void setArnoldiProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);
    void setGramSchmidtProjection(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);

    void setLbfgsSecantLeftPreconditioner(size_t aSecantStorageSize);
    void setLdfpSecantLeftPreconditioner(size_t aSecantStorageSize);
    void setLsr1SecantLeftPreconditioner(size_t aSecantStorageSize);
    void setSr1SecantLeftPreconditioner();
    void setBfgsSecantLeftPreconditioner();
    void setBarzilaiBorweinSecantLeftPreconditioner();

    void setRightPreconditioner(dotk::types::right_prec_t aType);

    virtual const std::shared_ptr<dotk::DOTk_OrthogonalProjection> & getProjection() const;
    virtual const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual const std::shared_ptr<dotk::DOTk_RightPreconditioner> & getRightPrec() const;

    virtual const std::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getRightPrecTimesVector() const;

private:
    std::shared_ptr<dotk::DOTk_LeftPreconditioner> m_LeftPreconditioner;
    std::shared_ptr<dotk::DOTk_RightPreconditioner> m_RightPreconditioner;
    std::shared_ptr<dotk::DOTk_OrthogonalProjection> m_OrthogonalProjection;

    std::shared_ptr<dotk::Vector<Real> > m_LeftPrecTimesResidual;
    std::shared_ptr<dotk::Vector<Real> > m_RightPrecTimesResidual;

private:
    void allocate(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);

private:
    DOTk_PrecGenMinResDataMng(const dotk::DOTk_PrecGenMinResDataMng &);
    dotk::DOTk_PrecGenMinResDataMng & operator=(const dotk::DOTk_PrecGenMinResDataMng &);
};

}

#endif /* DOTK_PRECGENMINRESDATAMNG_HPP_ */
