/*
 * DOTk_MexArrayPtr.cpp
 *
 *  Created on: Apr 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <mex.h>
#include "DOTk_MexArrayPtr.hpp"

namespace dotk
{

DOTk_MexArrayPtr::DOTk_MexArrayPtr() :
        m_Ptr(NULL)
{
}

DOTk_MexArrayPtr::DOTk_MexArrayPtr(mxArray* ptr_) :
        m_Ptr(ptr_)
{
}

void DOTk_MexArrayPtr::reset(mxArray* ptr_)
{
    if(m_Ptr != NULL)
    {
        mxDestroyArray(m_Ptr);
    }
    m_Ptr = ptr_;
}

mxArray* DOTk_MexArrayPtr::get() const
{
    return (m_Ptr);
}

mxArray* DOTk_MexArrayPtr::release()
{
    mxArray* ptr = m_Ptr;
    m_Ptr = NULL;
    return (ptr);
}

DOTk_MexArrayPtr::~DOTk_MexArrayPtr()
{
    if(m_Ptr != NULL)
    {
        mxDestroyArray(m_Ptr);
    }
    m_Ptr = NULL;
}

}
