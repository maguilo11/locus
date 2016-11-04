/*
 * DOTk_Variable.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#include <sstream>

#include "DOTk_OmpArray.hpp"
#include "DOTk_OmpArray.cpp"
#include "DOTk_OmpVector.hpp"
#include "DOTk_OmpVector.cpp"
#include "DOTk_MpiArray.hpp"
#include "DOTk_MpiArray.cpp"
#include "DOTk_MpiVector.hpp"
#include "DOTk_MpiVector.cpp"
#include "DOTk_MpiX_Array.hpp"
#include "DOTk_MpiX_Array.cpp"
#include "DOTk_MpiX_Vector.hpp"
#include "DOTk_MpiX_Vector.cpp"
#include "DOTk_SerialArray.hpp"
#include "DOTk_SerialArray.cpp"
#include "DOTk_SerialVector.hpp"
#include "DOTk_SerialVector.cpp"
#include "DOTk_Variable.hpp"


namespace dotk
{

DOTk_Variable::DOTk_Variable(dotk::types::variable_t type_) :
        m_Type(type_),
        m_Data(),
        m_LowerBound(),
        m_UpperBound()
{
}

DOTk_Variable::DOTk_Variable(dotk::types::variable_t type_, const dotk::vector<Real> & data_) :
        m_Type(type_),
        m_Data(data_.clone()),
        m_LowerBound(),
        m_UpperBound()
{
    this->initialize(data_);
}

DOTk_Variable::DOTk_Variable(dotk::types::variable_t type_,
                             const dotk::vector<Real> & data_,
                             const dotk::vector<Real> & lower_bound_,
                             const dotk::vector<Real> & upper_bound_) :
        m_Type(type_),
        m_Data(data_.clone()),
        m_LowerBound(lower_bound_.clone()),
        m_UpperBound(upper_bound_.clone())
{
    this->initialize(data_, lower_bound_, upper_bound_);
}

DOTk_Variable::~DOTk_Variable()
{
}

size_t DOTk_Variable::size() const
{
    return (m_Data->size());
}

dotk::types::variable_t DOTk_Variable::type() const
{
    return (m_Type);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Variable::data() const
{
    return (m_Data);
}

void DOTk_Variable::setLowerBound(Real value_)
{
    this->checkData();
    if(m_LowerBound.use_count() == 0)
    {
        m_LowerBound.reset();
        m_LowerBound = m_Data->clone();
    }
    m_LowerBound->fill(value_);
}

void DOTk_Variable::setLowerBound(const dotk::vector<Real> & lower_bound_)
{
    this->checkData();
    if(m_LowerBound.use_count() == 0)
    {
        m_LowerBound.reset();
        m_LowerBound = m_Data->clone();
    }
    m_LowerBound->copy(lower_bound_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Variable::lowerBound() const
{
    return (m_LowerBound);
}

void DOTk_Variable::setUpperBound(Real value_)
{
    this->checkData();
    if(m_UpperBound.use_count() == 0)
    {
        m_UpperBound.reset();
        m_UpperBound = m_Data->clone();
    }
    m_UpperBound->fill(value_);
}

void DOTk_Variable::setUpperBound(const dotk::vector<Real> & upper_bound_)
{
    this->checkData();
    if(m_UpperBound.use_count() == 0)
    {
        m_UpperBound.reset();
        m_UpperBound = m_Data->clone();
    }
    m_UpperBound->copy(upper_bound_);
}

const std::tr1::shared_ptr<dotk::vector<Real> > & DOTk_Variable::upperBound() const
{
    return (m_UpperBound);
}

void DOTk_Variable::allocateSerialArray(size_t size_, Real value_)
{
    m_Data.reset(new dotk::serial::array<Real>(size_, value_));
}

void DOTk_Variable::allocateMpiArray(MPI_Comm comm_, size_t size_, Real value_)
{
    m_Data.reset(new dotk::mpi::array<Real>(comm_, size_, value_));
}

void DOTk_Variable::allocateOmpArray(size_t size_, size_t num_threads_, Real value_)
{
    m_Data.reset(new dotk::omp::array<Real>(size_, num_threads_, value_));
}

void DOTk_Variable::allocateMpixArray(MPI_Comm comm_, size_t num_threads_, size_t size_, Real value_)
{
    m_Data.reset(new dotk::mpix::array<Real>(comm_, size_, num_threads_, value_));
}

void DOTk_Variable::allocateSerialVector(size_t size_, Real value_)
{
    m_Data.reset(new dotk::serial::vector<Real>(size_, value_));
}

void DOTk_Variable::allocateMpiVector(MPI_Comm comm_, size_t size_, Real value_)
{
    m_Data.reset(new dotk::mpi::vector<Real>(comm_, size_, value_));
}

void DOTk_Variable::allocateOmpVector(size_t size_, size_t num_threads_, Real value_)
{
    m_Data.reset(new dotk::omp::vector<Real>(size_, num_threads_, value_));
}

void DOTk_Variable::allocateMpixVector(MPI_Comm comm_, size_t num_threads_, size_t size_, Real value_)
{
    m_Data.reset(new dotk::mpix::vector<Real>(comm_, size_, num_threads_, value_));
}

void DOTk_Variable::checkData()
{
    if(m_Data.use_count() < 1)
    {
        std::perror("\n**** ERROR MESSAGE: DOTk_Variable::checkData() -> User did not allocate data for this dotk::Variable. ABORT ****\n");
        std::abort();
    }
}

void DOTk_Variable::initialize(const dotk::vector<Real> & data_)
{
    m_Data->copy(data_);
}

void DOTk_Variable::initialize(const dotk::vector<Real> & data_,
                               const dotk::vector<Real> & lower_bound_,
                               const dotk::vector<Real> & upper_bound_)
{
    try
    {
        size_t data_size = data_.size();
        size_t lower_bound_size = lower_bound_.size();
        if(lower_bound_size != data_size)
        {
            std::ostringstream msg;
            msg << "DOTk ERROR: DIMENSION MISMATCH BETWEEN LOWER BOUND AND DATA CONTAINERS."
                    << " DATA CONTAINER DIMENSION IS EQUAL TO " << data_size
                    << " AND LOWER BOUND CONTAINER DIMENSION IS EQUAL TO " << lower_bound_size << ": ABORT\n\n";
            throw msg.str().c_str();
        }

        size_t upper_bound_size = upper_bound_.size();
        if(upper_bound_size != data_size)
        {
            std::ostringstream msg;
            msg << "DOTk ERROR: DIMENSION MISMATCH BETWEEN UPPER BOUND AND DATA CONTAINERS."
                    << " DATA CONTAINER DIMENSION IS EQUAL TO " << data_size
                    << " AND UPPER BOUND CONTAINER DIMENSION IS EQUAL TO " << upper_bound_size << ": ABORT\n\n";
            throw msg.str().c_str();
        }

        for(size_t i = 0; i < lower_bound_.size(); ++ i)
        {
            if(lower_bound_[i] > upper_bound_[i])
            {
                std::ostringstream msg;
                msg << "DOTk ERROR: LOWER BOUND AT INDEX " << i << " EXCEEDS UPPER BOUND WITH VALUE " << lower_bound_[i]
                        << ". UPPER BOUND AT INDEX " << i << " HAS A VALUE OF " << upper_bound_[i] << ": ABORT\n\n";
                throw msg.str().c_str();
            }
        }

        m_Data->copy(data_);
        m_LowerBound->copy(lower_bound_);
        m_UpperBound->copy(upper_bound_);
    }
    catch(const char *error_msg)
    {
        std::cout << error_msg << std::flush;
    }
}

}

