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

DOTk_GmresTestMatrix::DOTk_GmresTestMatrix(const std::shared_ptr<dotk::DOTk_MultiVector<Real> > & aVector) :
        dotk::DOTk_LinearOperator(dotk::types::USER_DEFINED_MATRIX),
        mNumRows(4),
        mMatrix(std::make_shared<dotk::serial::DOTk_RowMatrix<Real>>(*aVector, mNumRows))
{
    this->allocate(aVector);
}

DOTk_GmresTestMatrix::~DOTk_GmresTestMatrix()
{
}

void DOTk_GmresTestMatrix::apply(const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                 const std::shared_ptr<dotk::Vector<Real> > & aMatrixTimesVector)
{
    mMatrix->matVec(*aVector, *aMatrixTimesVector);
}

void DOTk_GmresTestMatrix::apply(const std::shared_ptr<dotk::DOTk_OptimizationDataMng> & aMng,
                                 const std::shared_ptr<dotk::Vector<Real> > & aVector,
                                 const std::shared_ptr<dotk::Vector<Real> > & aMatrixTimesVector)
{
    mMatrix->matVec(*aVector, *aMatrixTimesVector);
}

void DOTk_GmresTestMatrix::allocate(const std::shared_ptr<dotk::DOTk_MultiVector<Real> > & aVector)
{
    // SET BLOCK CONTROL-CONTROL
    std::shared_ptr<dotk::Vector<Real> > dummy = aVector->clone();
    (*dummy->control())[0] = 0.218959186328090;
    (*dummy->control())[1] = 0.934692895940828;
    mMatrix->basis(0)->control()->update(1., *dummy->control(), 0.);
    (*dummy->control())[0] = 0.047044616214486;
    (*dummy->control())[1] = 0.383502077489860;
    mMatrix->basis(1)->control()->update(1., *dummy->control(), 0.);

    // SET BLOCK CONTROL-DUAL
    (*dummy->dual())[0] = 0.034572110527461;
    (*dummy->dual())[1] = 0.007698186211147;
    mMatrix->basis(0)->dual()->update(1., *dummy->dual(), 0.);
    (*dummy->dual())[0] = 0.053461635044525;
    (*dummy->dual())[1] = 0.383415650754895;
    mMatrix->basis(1)->dual()->update(1., *dummy->dual(), 0.);

    // SET BLOCK DUAL-CONTROL
    (*dummy->control())[0] = 0.678864716868319;
    (*dummy->control())[1] = 0.519416372067955;
    mMatrix->basis(2)->control()->update(1., *dummy->control(), 0.);
    (*dummy->control())[0] = 0.679296405836612;
    (*dummy->control())[1] = 0.830965346112366;
    mMatrix->basis(3)->control()->update(1., *dummy->control(), 0.);

    // SET BLOCK DUAL-DUAL
    (*dummy->dual())[0] = 0.529700193335163;
    (*dummy->dual())[1] = 0.066842237518561;
    mMatrix->basis(2)->dual()->update(1., *dummy->dual(), 0.);
    (*dummy->dual())[0] = 0.671149384077242;
    (*dummy->dual())[1] = 0.417485974457807;
    mMatrix->basis(3)->dual()->update(1., *dummy->dual(), 0.);
}

}
