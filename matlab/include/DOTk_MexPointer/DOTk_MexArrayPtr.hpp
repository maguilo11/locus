/*
 * DOTk_MexArrayPtr.hpp
 *
 *  Created on: Apr 12, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_MEXARRAYPTR_HPP_
#define DOTK_MEXARRAYPTR_HPP_

#include <mex.h>

namespace dotk
{

class DOTk_MexArrayPtr
{
public:
    DOTk_MexArrayPtr();
    explicit DOTk_MexArrayPtr(mxArray* ptr_);
    ~DOTk_MexArrayPtr();

    void reset(mxArray* ptr_);
    mxArray* get() const;
    mxArray* release();

private:
    mxArray* m_Ptr;

private:
    DOTk_MexArrayPtr(const dotk::DOTk_MexArrayPtr& rhs_);
    dotk::DOTk_MexArrayPtr& operator=(const dotk::DOTk_MexArrayPtr& rhs_);
};

}

#endif /* DOTK_MEXARRAYPTR_HPP_ */
