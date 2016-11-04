/*
 * DOTk_GmresTestMatrix.cpp
 *
 *  Created on: Nov 2, 2014
 *      Author: Miguel A. Aguilo Valentin
 */

#include "DOTk_RowMatrix.hpp"
#include "DOTk_RowMatrix.cpp"
#include "DOTk_MultiVector.hpp"
#include "DOTk_MultiVector.cpp"
#include "DOTk_GmresTestMatrix.hpp"
#include "DOTk_OptimizationDataMng.hpp"

namespace dotk
{

DOTk_GmresTestMatrix::DOTk_GmresTestMatrix(const std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> > & vector_) :
        dotk::DOTk_LinearOperator(dotk::types::USER_DEFINED_MATRIX),
        m_Matrix(new dotk::serial::DOTk_RowMatrix<Real>(*vector_, 4))
{
    this->allocate(vector_);
}

DOTk_GmresTestMatrix::~DOTk_GmresTestMatrix()
{
}

void DOTk_GmresTestMatrix::apply(const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_)
{
    m_Matrix->matVec(*vector_, *matrix_times_vector_);
}

void DOTk_GmresTestMatrix::apply(const std::tr1::shared_ptr<dotk::DOTk_OptimizationDataMng> & mng_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & vector_,
                                 const std::tr1::shared_ptr<dotk::vector<Real> > & matrix_times_vector_)
{
    m_Matrix->matVec(*vector_, *matrix_times_vector_);
}

void DOTk_GmresTestMatrix::allocate(const std::tr1::shared_ptr<dotk::DOTk_MultiVector<Real> > & vector_)
{
    // SET BLOCK CONTROL-CONTROL
    std::tr1::shared_ptr<dotk::vector<Real> > dummy = vector_->clone();
    (*dummy->control())[0] = 0.218959186328090;
    (*dummy->control())[1] = 0.934692895940828;
    m_Matrix->basis(0)->control()->copy(*dummy->control());
    (*dummy->control())[0] = 0.047044616214486;
    (*dummy->control())[1] = 0.383502077489860;
    m_Matrix->basis(1)->control()->copy(*dummy->control());

    // SET BLOCK CONTROL-DUAL
    (*dummy->dual())[0] = 0.034572110527461;
    (*dummy->dual())[1] = 0.007698186211147;
    m_Matrix->basis(0)->dual()->copy(*dummy->dual());
    (*dummy->dual())[0] = 0.053461635044525;
    (*dummy->dual())[1] = 0.383415650754895;
    m_Matrix->basis(1)->dual()->copy(*dummy->dual());

    // SET BLOCK DUAL-CONTROL
    (*dummy->control())[0] = 0.678864716868319;
    (*dummy->control())[1] = 0.519416372067955;
    m_Matrix->basis(2)->control()->copy(*dummy->control());
    (*dummy->control())[0] = 0.679296405836612;
    (*dummy->control())[1] = 0.830965346112366;
    m_Matrix->basis(3)->control()->copy(*dummy->control());

    // SET BLOCK DUAL-DUAL
    (*dummy->dual())[0] = 0.529700193335163;
    (*dummy->dual())[1] = 0.066842237518561;
    m_Matrix->basis(2)->dual()->copy(*dummy->dual());
    (*dummy->dual())[0] = 0.671149384077242;
    (*dummy->dual())[1] = 0.417485974457807;
    m_Matrix->basis(3)->dual()->copy(*dummy->dual());
}

}
