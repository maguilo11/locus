/*
 * TRROM_MxArrayPtr.hpp
 *
 *  Created on: Nov 12, 2016
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef TRROM_MATLAB_INCLUDE_TRROM_MXARRAYPTR_HPP_
#define TRROM_MATLAB_INCLUDE_TRROM_MXARRAYPTR_HPP_

#include <mex.h>

namespace trrom
{

class MxArrayPtr
{
public:
    MxArrayPtr() :
            m_Ptr(nullptr)
    {
        /*!
         * Null pointer constructor for class MxArrayPtr, m_Ptr = nullptr.
         */
    }
    explicit MxArrayPtr(mxArray* ptr_) :
            m_Ptr(ptr_)
    {
        /*!
         * Main constructor for class MxArrayPtr, set member data m_Ptr = ptr_
         */
    }
    ~MxArrayPtr()
    {
        /*!
         * Destructor for class MxArrayPtr, deallocate memory for member pointer m_Ptr
         */
        if(m_Ptr != nullptr)
        {
            mxDestroyArray(m_Ptr);
        }
        m_Ptr = nullptr;
    }
    void reset(mxArray* ptr_)
    {
        /*!
         * Reset memory of member data to input pointer
         */
        if(m_Ptr != nullptr)
        {
            mxDestroyArray(m_Ptr);
        }
        m_Ptr = ptr_;
    }
    mxArray* get() const
    {
        /*!
         * Get member data pointer
         */
        return (m_Ptr);
    }
    mxArray* release()
    {
        /*!
         * Release memory of private pointer
         */
        mxArray* ptr = m_Ptr;
        m_Ptr = nullptr;
        return (ptr);
    }

private:
    mxArray* m_Ptr;

private:
    MxArrayPtr(const trrom::MxArrayPtr& rhs_);
    trrom::MxArrayPtr& operator=(const trrom::MxArrayPtr& rhs_);
};

}

#endif /* TRROM_MATLAB_INCLUDE_TRROM_MXARRAYPTR_HPP_ */
