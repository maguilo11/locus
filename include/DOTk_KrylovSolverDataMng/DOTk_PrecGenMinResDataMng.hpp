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

template<class T>
class vector;

class DOTk_PrecGenMinResDataMng : public dotk::DOTk_KrylovSolverDataMng
{
public:
    DOTk_PrecGenMinResDataMng(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_,
                              const std::tr1::shared_ptr<dotk::DOTk_LinearOperator> & operator_,
                              size_t max_num_itr_ = 200);
    virtual ~DOTk_PrecGenMinResDataMng();

    void setDirectSolver(dotk::types::direct_solver_t type_);
    void setArnoldiProjection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);
    void setGramSchmidtProjection(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

    void setLbfgsSecantLeftPreconditioner(size_t secant_storage_);
    void setLdfpSecantLeftPreconditioner(size_t secant_storage_);
    void setLsr1SecantLeftPreconditioner(size_t secant_storage_);
    void setSr1SecantLeftPreconditioner();
    void setBfgsSecantLeftPreconditioner();
    void setBarzilaiBorweinSecantLeftPreconditioner();

    void setRightPreconditioner(dotk::types::right_prec_t type_);

    virtual const std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> & getProjection() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual const std::tr1::shared_ptr<dotk::DOTk_RightPreconditioner> & getRightPrec() const;

    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getLeftPrecTimesVector() const;
    virtual const std::tr1::shared_ptr<dotk::vector<Real> > & getRightPrecTimesVector() const;

private:
    std::tr1::shared_ptr<dotk::DOTk_LeftPreconditioner> m_LeftPreconditioner;
    std::tr1::shared_ptr<dotk::DOTk_RightPreconditioner> m_RightPreconditioner;
    std::tr1::shared_ptr<dotk::DOTk_OrthogonalProjection> m_OrthogonalProjection;

    std::tr1::shared_ptr<dotk::vector<Real> > m_LeftPrecTimesResidual;
    std::tr1::shared_ptr<dotk::vector<Real> > m_RightPrecTimesResidual;

private:
    void allocate(const std::tr1::shared_ptr<dotk::DOTk_Primal> & primal_);

private:
    DOTk_PrecGenMinResDataMng(const dotk::DOTk_PrecGenMinResDataMng &);
    dotk::DOTk_PrecGenMinResDataMng & operator=(const dotk::DOTk_PrecGenMinResDataMng &);
};

}

#endif /* DOTK_PRECGENMINRESDATAMNG_HPP_ */
