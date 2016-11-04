/*
 * DOTk_Variable.hpp
 *
 *  Created on: Feb 18, 2015
 *      Author: Miguel A. Aguilo Valentin
 */

#ifndef DOTK_VARIABLE_HPP_
#define DOTK_VARIABLE_HPP_

#include <mpi.h>
#include "DOTk_Types.hpp"

namespace dotk
{

template<class T>
class vector;

class DOTk_Variable
{
public:
    explicit DOTk_Variable(dotk::types::variable_t type_);
    DOTk_Variable(dotk::types::variable_t type_, const dotk::vector<Real> & data_);
    DOTk_Variable(dotk::types::variable_t type_,
                  const dotk::vector<Real> & data_,
                  const dotk::vector<Real> & lower_bound_,
                  const dotk::vector<Real> & upper_bound_);
    ~DOTk_Variable();

    size_t size() const;
    dotk::types::variable_t type() const;
    const std::tr1::shared_ptr<dotk::vector<Real> > & data() const;

    void setLowerBound(Real value_);
    void setLowerBound(const dotk::vector<Real> & lower_bound_);
    const std::tr1::shared_ptr<dotk::vector<Real> > & lowerBound() const;

    void setUpperBound(Real value_);
    void setUpperBound(const dotk::vector<Real> & upper_bound_);
    const std::tr1::shared_ptr<dotk::vector<Real> > & upperBound() const;

    void allocateSerialArray(size_t size_, Real value_);
    void allocateMpiArray(MPI_Comm comm_, size_t size_, Real value_);
    void allocateOmpArray(size_t size_, size_t num_threads_, Real value_);
    void allocateMpixArray(MPI_Comm comm_, size_t num_threads_, size_t size_, Real value_);

    void allocateSerialVector(size_t size_, Real value_);
    void allocateMpiVector(MPI_Comm comm_, size_t size_, Real value_);
    void allocateOmpVector(size_t size_, size_t num_threads_, Real value_);
    void allocateMpixVector(MPI_Comm comm_, size_t num_threads_, size_t size_, Real value_);

private:
    void checkData();
    void initialize(const dotk::vector<Real> & data_);
    void initialize(const dotk::vector<Real> & data_,
                    const dotk::vector<Real> & lower_bound_,
                    const dotk::vector<Real> & upper_bound_);

private:
    dotk::types::variable_t m_Type;
    std::tr1::shared_ptr<dotk::vector<Real> > m_Data;
    std::tr1::shared_ptr<dotk::vector<Real> > m_LowerBound;
    std::tr1::shared_ptr<dotk::vector<Real> > m_UpperBound;

private:
    DOTk_Variable(const dotk::DOTk_Variable &);
    dotk::DOTk_Variable & operator=(const dotk::DOTk_Variable &);
};

}

#endif /* DOTK_VARIABLE_HPP_ */
