/*
 * TRROM_EpetraVector.hpp
 *
 *  Created on: Sep 30, 2016
 *      Author: maguilo
 */

#ifndef TRROM_EPETRAVECTOR_HPP_
#define TRROM_EPETRAVECTOR_HPP_

#include <mpi.h>
#include <tr1/memory>

#include "Epetra_Map.h"
#include "Epetra_config.h"
#include "Epetra_Vector.h"
#include "Epetra_MpiComm.h"

#include "TRROM_Vector.hpp"

namespace trrom
{

class EpetraVector : public trrom::Vector<double>
{
public:
    explicit EpetraVector(const Epetra_BlockMap & map_) :
            m_Data(std::make_shared<Epetra_Vector>(map_))
    {
    }
    EpetraVector(Epetra_DataAccess copy_or_view_access_, const Epetra_MultiVector & source_, int index_) :
            m_Data(std::make_shared<Epetra_Vector>(copy_or_view_access_, source_, index_))
    {
    }
    virtual ~EpetraVector()
    {
    }

    void scale(const double & alpha_)
    {
        m_Data->Scale(alpha_);
    }
    double min(int & index_) const
    {
        double output = 0;
        m_Data->MinValue(&output);
        return (output);
    }
    double max(int & index_) const
    {
        double output = 0;
        m_Data->MaxValue(&output);
        return (output);
    }
    double dot(const trrom::Vector<double> & input_) const
    {
        /** Returns global dot product between two vectors */
        assert(input_.size() == this->size());

        /** Returns global sum of vector */
        double local_sum = 0.;
        int local_length = m_Data->MyLength();
        for(int index = 0; index < local_length; ++index)
        {
            local_sum += input_[index] * m_Data->operator [](index);
        }

        int count = 1;
        double global_sum = 0.;
        m_Data->Comm().SumAll(&local_sum, &global_sum, count);

        return (global_sum);
    }
    double sum() const
    {
        /** Returns global sum of vector */
        double local_sum = 0.;
        int local_length = m_Data->MyLength();
        for(int index = 0; index < local_length; ++index)
        {
            local_sum += (*m_Data)[index];
        }

        int count = 1;
        double global_sum = 0.;
        m_Data->Comm().SumAll(&local_sum, &global_sum, count);

        return (global_sum);
    }
    double norm() const
    {
        /** Returns global Euclidean norm of vector */
        double output = 0;
        m_Data->Norm2(&output);
        return (output);
    }
    void modulus()
    {
        /** Puts element-wise absolute values */
        m_Data->Abs(*m_Data);
    }
    void fill(const double & input_)
    {
        /** Fills vector with input double value */
        m_Data->PutScalar(input_);
    }

    void update(const double & alpha_, const trrom::Vector<double> & input_, const double & beta_)
    {
        /** Update multi-vector values with scaled values of A, this = ScalarThis*this + ScalarA*A. */
        assert(this->size() == input_.size());
        const trrom::EpetraVector & input = dynamic_cast<const trrom::EpetraVector &>(input_);
        m_Data->Update(alpha_, *input.m_Data, beta_);
    }
    void elementWiseMultiplication(const trrom::Vector<double> & input_)
    {
        /** **************************************************************************************
         * Component wise multiplication. For Epetra_Vectors this calculation is defined as
         * this = ScalarThis * this + ScalarAB * B @ A where @ denotes element-wise multiplication.
         * *************************************************************************************** */
        assert(this->size() == input_.size());

        double this_scalar = 0.;
        double multiplication_scalar = 1.;
        const trrom::EpetraVector & input = dynamic_cast<const trrom::EpetraVector &>(input_);
        m_Data->Multiply(multiplication_scalar, *m_Data, *input.m_Data, this_scalar);
    }
    int size() const
    {
        /** Returns local number of entries */
        return (m_Data->MyLength());
    }
    double & operator [](int index_)
    {
        /** Operator overloads square bracket operator */
        return (m_Data->operator [](index_));
    }
    const double & operator [](int index_) const
    {
        /** Operator overloads square bracket operator */
        return (m_Data->operator [](index_));
    }
    std::shared_ptr<trrom::Vector<double> > create(int global_length = 0) const
    {
        /*! Creates copy of this vector */
        std::shared_ptr<trrom::EpetraVector> this_copy;
        if(global_length == 0)
        {
            this_copy = std::make_shared<trrom::EpetraVector>(m_Data->Map());
        }
        else
        {
            const int index_base = m_Data->Map().IndexBase();
            const int element_size = m_Data->Map().ElementSize();
            Epetra_BlockMap map(global_length, element_size, index_base, m_Data->Comm());
            this_copy = std::make_shared<trrom::EpetraVector>(map);
        }
        return (this_copy);
    }
    int getNumProc() const
    {
        return (m_Data->Comm().NumProc());
    }
    int getNumGlobalElements() const
    {
        return (m_Data->Map().NumGlobalElements());
    }
    std::shared_ptr<Epetra_Vector> & data()
    {
        return (m_Data);
    }
    const std::shared_ptr<Epetra_Vector> & data() const
    {
        return (m_Data);
    }

private:
    std::shared_ptr<Epetra_Vector> m_Data;

private:
    EpetraVector(const trrom::EpetraVector & that_);
    trrom::EpetraVector & operator=(const trrom::EpetraVector &);
};

}

#endif /* TRROM_EPETRAVECTOR_HPP_ */
