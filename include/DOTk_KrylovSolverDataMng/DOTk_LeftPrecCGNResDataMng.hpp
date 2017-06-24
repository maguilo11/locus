/*
 * DOTk_LeftPrecCGNResDataMng.hpp
 *
 *  Created on: Nov 3, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_LEFTPRECCGNRESDATAMNG_HPP_
#define DOTK_LEFTPRECCGNRESDATAMNG_HPP_

#include "DOTk_KrylovSolverDataMng.hpp"

namespace dotk
{

class DOTk_Primal;
class DOTk_LinearOperator;
class DOTk_LeftPreconditioner;

template<typename ScalarType>
class Vector;

class DOTk_LeftPrecCGNResDataMng : public dotk::DOTk_KrylovSolverDataMng
{
public:
    DOTk_LeftPrecCGNResDataMng(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal,
                               const std::shared_ptr<dotk::DOTk_LinearOperator> & aLinearOperator);
    virtual ~DOTk_LeftPrecCGNResDataMng();

    void setLbfgsSecantLeftPreconditioner(size_t aSecantStorageSize);
    void setLdfpSecantLeftPreconditioner(size_t aSecantStorageSize);
    void setLsr1SecantLeftPreconditioner(size_t aSecantStorageSize);
    void setSr1SecantLeftPreconditioner();
    void setBfgsSecantLeftPreconditioner();
    void setBarzilaiBorweinSecantLeftPreconditioner();

    virtual const std::shared_ptr<dotk::DOTk_LeftPreconditioner> & getLeftPrec() const;
    virtual const std::shared_ptr<dotk::Vector<Real> > & getLeftPrecTimesVector() const;

private:
    std::shared_ptr<dotk::DOTk_LeftPreconditioner> m_LeftPreconditioner;
    std::shared_ptr<dotk::Vector<Real> > m_LeftPrecTimesResidual;

private:
    void allocate(const std::shared_ptr<dotk::DOTk_Primal> & aPrimal);

private:
    DOTk_LeftPrecCGNResDataMng(const dotk::DOTk_LeftPrecCGNResDataMng &);
    dotk::DOTk_LeftPrecCGNResDataMng & operator=(const dotk::DOTk_LeftPrecCGNResDataMng &);
};

}

#endif /* DOTK_LEFTPRECCGNRESDATAMNG_HPP_ */
