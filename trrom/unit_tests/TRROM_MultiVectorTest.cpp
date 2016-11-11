/*
 * TRROM_MultiVectorTest.cpp
 *
 *  Created on: Oct 27, 2016
 *      Author: maguilo
 */

#include <gtest/gtest.h>

#include <Epetra_LocalMap.h>

#include "TRROM_Matrix.hpp"
#include "TRROM_EpetraVector.hpp"

namespace trrom
{

template<typename ScalarType>
class MatrixBase
{
public:
    virtual ~MatrixBase()
    {
    }
    virtual int getNumRows() const = 0;
    virtual int getNumCols() const = 0;
    virtual void fill(ScalarType value_) = 0;
    virtual void scale(ScalarType value_) = 0;
    virtual void update(const ScalarType & input_scalar_,
                        const trrom::MatrixBase<ScalarType> & input_,
                        const ScalarType & this_scalar_) = 0;
    virtual void gemv(const bool & this_transpose_,
                      const ScalarType & this_scalar_,
                      const trrom::Vector<ScalarType> & input_,
                      const ScalarType & output_scalar_,
                      trrom::Vector<ScalarType> & output_) const = 0;
    virtual void gemm(const bool & this_transpose_,
                      const bool & input_transpose_,
                      const double & this_scalar_,
                      const trrom::MatrixBase<double> & input_,
                      const double & output_scalar_,
                      trrom::MatrixBase<double> & output_) const = 0;
    virtual void replaceLocalValue(const int & row_index_, const int & column_index_, const ScalarType & value_) = 0;
    virtual void replaceGlobalValue(const int & row_index_, const int & column_index_, const ScalarType & value_) = 0;
};

template<typename ScalarType>
class MultiVector : public trrom::MatrixBase<ScalarType>
{
public:
    virtual ~MultiVector()
    {
    }
    virtual int getNumRows() const = 0;
    virtual int getNumCols() const = 0;
    virtual void fill(ScalarType value_) = 0;
    virtual void scale(ScalarType value_) = 0;
    virtual void update(const ScalarType & input_scalar_,
                        const trrom::MatrixBase<ScalarType> & input_,
                        const ScalarType & this_scalar_) = 0;
    virtual void gemv(const bool & this_transpose_,
                      const ScalarType & this_scalar_,
                      const trrom::Vector<ScalarType> & input_,
                      const ScalarType & output_scalar_,
                      trrom::Vector<ScalarType> & output_) const = 0;
    virtual void gemm(const bool & this_transpose_,
                      const bool & input_transpose_,
                      const double & this_scalar_,
                      const trrom::MatrixBase<double> & input_,
                      const double & output_scalar_,
                      trrom::MatrixBase<double> & output_) const = 0;
    virtual void replaceLocalValue(const int & row_index_, const int & column_index_, const ScalarType & value_) = 0;
    virtual void replaceGlobalValue(const int & row_index_, const int & column_index_, const ScalarType & value_) = 0;

    virtual void insert(const trrom::Vector<ScalarType> & input_) = 0;
    virtual const std::tr1::shared_ptr<trrom::Vector<ScalarType> > & getVector(int global_vector_index_) = 0;
    virtual std::tr1::shared_ptr<trrom::MultiVector<ScalarType> > create() const = 0;
    virtual std::tr1::shared_ptr<trrom::MultiVector<ScalarType> > create(int global_num_elements_, int global_num_vectors_) const = 0;
};

class EpetraMultiVector : public trrom::MultiVector<double>
{
public:
    EpetraMultiVector(const Epetra_BlockMap & map_, int num_vectors_, bool init_ensemble_ = false) :
            m_NumVecStored(0),
            m_ThisVector(),
            m_CurrentEnsemble(),
            m_DataWarehouse(new Epetra_MultiVector(map_, num_vectors_))
    {
        if(init_ensemble_ == true)
        {
            m_CurrentEnsemble.reset(new Epetra_MultiVector(map_, num_vectors_));
        }
    }
    virtual ~EpetraMultiVector()
    {
    }

    int getNumRows() const
    {
        return (m_CurrentEnsemble->GlobalLength());
    }
    int getNumCols() const
    {
        return (m_CurrentEnsemble->NumVectors());
    }
    void fill(double value_)
    {
        if(m_CurrentEnsemble.use_count() <= 0)
        {
            m_NumVecStored = 1;
            m_CurrentEnsemble.reset(new Epetra_MultiVector(m_DataWarehouse->Map(), m_NumVecStored));
        }
        m_CurrentEnsemble->PutScalar(value_);
    }
    void scale(double value_)
    {
        if(m_CurrentEnsemble.use_count() <= 0)
        {
            m_NumVecStored = 1;
            m_CurrentEnsemble.reset(new Epetra_MultiVector(m_DataWarehouse->Map(), m_NumVecStored));
        }
        m_CurrentEnsemble->Scale(value_);
    }
    void update(const double & input_scalar_, const trrom::MatrixBase<double> & input_, const double & this_scalar_)
    {
        try
        {
            if(this->getNumCols() != input_.getNumCols())
            {
                std::ostringstream error;
                error << "\nERROR IN: " << __FILE__ << ", LINE: "
                      << __LINE__ << " -> Input matrix and this->matrix have different number of columns.\n";
                throw error.str().c_str();
            }
            if(this->getNumRows() != input_.getNumRows())
            {
                std::ostringstream error;
                error << "\nERROR IN: " << __FILE__ << ", LINE: "
                      << __LINE__ << " -> Input matrix and this->matrix have different number of rows.\n";
                throw error.str().c_str();
            }
            const trrom::EpetraMultiVector & input = dynamic_cast<const trrom::EpetraMultiVector &>(input_);
            this->m_CurrentEnsemble->Update(input_scalar_, *input.data(), this_scalar_);
        }
        catch(const char *error_msg)
        {
            std::cout << error_msg << std::flush;
        }
    }
    void gemv(const bool & this_transpose_,
              const double & this_scalar_,
              const trrom::Vector<double> & input_,
              const double & output_scalar_,
              trrom::Vector<double> & output_) const
    {
        /*! output = output_scalar * output + input_scalar * this_matrix @ input_vector, where @ denotes element-wise multiplication
         *
         */
        try
        {
            if(m_CurrentEnsemble.use_count() <= 0)
            {
                std::ostringstream error;
                error << "\nERROR IN: "
                      << __FILE__ << ", LINE: "
                      << __LINE__ << " -> EpetraMultiVector data is not initialize. Use fill(input) before calling gemv.\n";
                throw error.str().c_str();
            }
            std::vector<char> this_transpose = this->transpose(this_transpose_);
            trrom::EpetraVector & output = dynamic_cast<trrom::EpetraVector &>(output_);
            const trrom::EpetraVector & input = dynamic_cast<const trrom::EpetraVector &>(input_);
            output.data()->Multiply(this_transpose[0], 'N', this_scalar_, *m_CurrentEnsemble, *input.data(), output_scalar_);
        }
        catch(const char *error_msg)
        {
            std::cout << error_msg << std::flush;
        }
    }
    void gemm(const bool & this_transpose_,
              const bool & input_transpose_,
              const double & this_scalar_,
              const trrom::MatrixBase<double> & input_,
              const double & output_scalar_,
              trrom::MatrixBase<double> & output_) const
    {
        try
        {
            if(m_CurrentEnsemble.use_count() <= 0)
            {
                std::ostringstream error;
                error << "\nERROR IN: "
                      << __FILE__ << ", LINE: "
                      << __LINE__ << " -> EpetraMultiVector data is not initialize. Use insert(input) before calling gemm.\n";
                throw error.str().c_str();
            }
            trrom::EpetraMultiVector & output = dynamic_cast<trrom::EpetraMultiVector &>(output_);
            const trrom::EpetraMultiVector & input = dynamic_cast<const trrom::EpetraMultiVector &>(input_);
            std::vector<char> this_transpose = this->transpose(this_transpose_);
            std::vector<char> input_transpose = this->transpose(input_transpose_);
            output.data()->Multiply(this_transpose[0],
                                    input_transpose[0],
                                    this_scalar_,
                                    *m_CurrentEnsemble,
                                    *input.data(),
                                    output_scalar_);
        }
        catch(const char *error_msg)
        {
            std::cout << error_msg << std::flush;
        }
    }
    void replaceLocalValue(const int & row_index_, const int & vector_index_, const double & value_)
    {
        m_CurrentEnsemble->ReplaceMyValue(row_index_, vector_index_, value_);
    }
    void replaceGlobalValue(const int & global_row_index_, const int & vector_index_, const double & value_)
    {
        m_CurrentEnsemble->ReplaceGlobalValue(global_row_index_, vector_index_, value_);
    }

    void insert(const trrom::Vector<double> & input_)
    {
        int ensemble_size = m_DataWarehouse->NumVectors();
        const trrom::EpetraVector & input = dynamic_cast<const trrom::EpetraVector &>(input_);
        if(m_NumVecStored < ensemble_size)
        {
            m_DataWarehouse->operator()(m_NumVecStored)->Update(1., *input.data(), 0.);
            int start_index = 0;
            int global_num_vectors = m_NumVecStored + 1;
            m_CurrentEnsemble.reset(new Epetra_MultiVector(Epetra_DataAccess::Copy,
                                                           *m_DataWarehouse,
                                                           start_index,
                                                           global_num_vectors));
            m_NumVecStored++;
        }
        else
        {
            Epetra_MultiVector copy(*m_DataWarehouse);
            int index_base = 0;
            int element_size = 1;
            int global_num_elements = m_DataWarehouse->GlobalLength();
            int new_global_num_vectors = 2 * m_DataWarehouse->NumVectors();
            Epetra_BlockMap map(global_num_elements, element_size, index_base, m_DataWarehouse->Comm());
            m_DataWarehouse.reset(new Epetra_MultiVector(map, new_global_num_vectors));
            for(int index = 0; index < copy.NumVectors(); ++index)
            {
                m_DataWarehouse->operator()(index)->Update(1., *copy.operator()(index), 0.);
            }
            m_DataWarehouse->operator()(m_NumVecStored)->Update(1., *input.data(), 0.);

            int start_index = 0;
            int global_num_vectors = m_NumVecStored + 1;
            m_CurrentEnsemble.reset(new Epetra_MultiVector(Epetra_DataAccess::Copy,
                                                           *m_DataWarehouse,
                                                           start_index,
                                                           global_num_vectors));
            m_NumVecStored++;
        }
    }
    const std::tr1::shared_ptr< trrom::Vector<double> > & getVector(int global_vector_index_)
    {
        m_ThisVector.reset(new trrom::EpetraVector(Epetra_DataAccess::View, *m_CurrentEnsemble, global_vector_index_));
        return (m_ThisVector);
    }
    std::tr1::shared_ptr<trrom::MultiVector<double> > create() const
    {
        /** Creates copy of this vector */
        const int global_num_vectors = this->getNumCols();
        const Epetra_BlockMap & map = m_CurrentEnsemble->Map();
        std::tr1::shared_ptr<trrom::EpetraMultiVector> this_copy(new trrom::EpetraMultiVector(map, global_num_vectors));
        return (this_copy);
    }
    std::tr1::shared_ptr<trrom::MultiVector<double> > create(int global_num_elements_, int num_vectors_) const
    {
        /** Creates copy of this vector with user supplied dimensions */
        const int index_base = m_CurrentEnsemble->Map().IndexBase();
        const int element_size = m_CurrentEnsemble->Map().ElementSize();
        Epetra_BlockMap map(global_num_elements_, element_size, index_base, m_CurrentEnsemble->Comm());
        std::tr1::shared_ptr<trrom::EpetraMultiVector> new_copy(new trrom::EpetraMultiVector(map, num_vectors_));
        return (new_copy);
    }

    int maxStorageSize()
    {
        return (m_DataWarehouse->NumVectors());
    }
    const std::tr1::shared_ptr<Epetra_MultiVector> & data() const
    {
        return (m_CurrentEnsemble);
    }

private:
    std::vector<char> transpose(const bool & input_) const
    {
        std::vector<char> output;
        if(input_ == true)
        {
            output.push_back('T');
        }
        else
        {
            output.push_back('N');
        }
        return (output);
    }

private:
    int m_NumVecStored;
    std::tr1::shared_ptr< trrom::Vector<double> > m_ThisVector;
    std::tr1::shared_ptr<Epetra_MultiVector> m_CurrentEnsemble;
    std::tr1::shared_ptr<Epetra_MultiVector> m_DataWarehouse;

private:
    EpetraMultiVector(const trrom::EpetraMultiVector &);
    trrom::EpetraMultiVector & operator=(const trrom::EpetraMultiVector &);
};

}

namespace MultiVectorTest
{

void checkResults(const trrom::Vector<double> & gold_, const trrom::Vector<double> & results_, double tolerance_ = 1e-6)
{
    assert(gold_.size() == results_.size());
    int num_local_elements = results_.size();
    for(int index = 0; index < num_local_elements; ++index)
    {
        EXPECT_NEAR(gold_[index], results_[index], tolerance_);
    }
}

void checkResults(const std::tr1::shared_ptr<trrom::EpetraMultiVector> & gold_,
                  const std::tr1::shared_ptr<trrom::EpetraMultiVector> & results_,
                  double tolerance_ = 1e-6)
{
    assert(gold_->getNumCols() == results_->getNumCols());
    assert(gold_->getNumRows() == results_->getNumRows());

    int number_vectors = results_->getNumCols();
    for(int vector_index = 0; vector_index < number_vectors; ++vector_index)
    {
        int local_number_elements = gold_->getVector(vector_index)->size();
        for(int element_index = 0; element_index < local_number_elements; ++element_index)
        {
            EXPECT_NEAR((*gold_->getVector(vector_index))[element_index],
                        (*results_->getVector(vector_index))[element_index],
                        tolerance_);
        }
    }
}

TEST(EpetraVector, test)
{
    Epetra_MpiComm comm(MPI_COMM_WORLD);

    int num_local_elements = 10;
    int num_proc = comm.NumProc();
    int global_num_elements = num_local_elements * num_proc;

    int index_base = 0;
    int element_size = 1;
    Epetra_BlockMap block_map(global_num_elements, element_size, index_base, comm);

    int global_num_vectors = 4;
    Epetra_MultiVector A(block_map, global_num_vectors);

    A.PutScalar(1.);
    std::vector<double> results(global_num_vectors, 0.);
    int error = A.Norm1(results.data());
    EXPECT_EQ(0, error);

    double tolerance = 1e-6;
    double gold = global_num_elements;
    for(int index = 0; index < global_num_vectors; ++index)
    {
        EXPECT_NEAR(gold, results[index], tolerance);
    }
}

TEST(EpetraMultiVector, insert)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);

    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(map, global_num_vectors);
    EXPECT_EQ(global_num_vectors, X.maxStorageSize());

    trrom::EpetraVector x(map);
    X.insert(x);
    EXPECT_EQ(1, X.getNumCols());
    EXPECT_EQ(global_num_elements, X.getNumRows());
}

TEST(EpetraMultiVector, getVector)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);

    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(map, global_num_vectors);
    EXPECT_EQ(global_num_vectors, X.maxStorageSize());

    trrom::EpetraVector x(map);
    x.fill(2);
    X.insert(x);

    std::tr1::shared_ptr< trrom::Vector<double> > gold = x.create();
    gold->fill(2);
    checkResults(*gold, *X.getVector(0));
}

TEST(EpetraMultiVector, fill)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);

    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(map, global_num_vectors);

    X.fill(1);
    EXPECT_EQ(1, X.getNumCols());
    EXPECT_EQ(global_num_elements, X.getNumRows());

    double value = 0;
    X.data()->Norm1(&value);
    double tolerance = 1e-6;
    EXPECT_NEAR(20, value, tolerance);
}

TEST(EpetraMultiVector, scale)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);

    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(map, global_num_vectors);

    X.fill(1);
    X.scale(2);
    EXPECT_EQ(1, X.getNumCols());
    EXPECT_EQ(global_num_elements, X.getNumRows());

    double value = 0;
    X.data()->Norm1(&value);
    double tolerance = 1e-6;
    EXPECT_NEAR(40, value, tolerance);
}

TEST(EpetraMultiVector, copy)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);

    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(map, global_num_vectors);
    EXPECT_EQ(global_num_vectors, X.maxStorageSize());

    X.fill(0);
    EXPECT_EQ(1, X.getNumCols());
    EXPECT_EQ(global_num_elements, X.getNumRows());

    trrom::EpetraVector x(map);
    X.insert(x);
    EXPECT_EQ(2, X.getNumCols());
    EXPECT_EQ(global_num_elements, X.getNumRows());

    std::vector<double> value(X.getNumCols(), -1.);
    X.data()->Norm1(value.data());
    double tolerance = 1e-6;
    EXPECT_NEAR(0, value[0], tolerance);
    EXPECT_NEAR(0, value[1], tolerance);

    trrom::EpetraMultiVector Y(map, global_num_vectors);
    EXPECT_EQ(global_num_vectors, Y.maxStorageSize());

    Y.insert(x);
    Y.insert(x);
    EXPECT_EQ(2, Y.getNumCols());
    EXPECT_EQ(global_num_elements, Y.getNumRows());

    Y.fill(2);
    value.assign(value.size(), 0.);
    Y.data()->Norm1(value.data());
    EXPECT_NEAR(40, value[0], tolerance);
    EXPECT_NEAR(40, value[1], tolerance);

    X.update(1., Y, 0.);
    value.assign(value.size(), 0.);
    X.data()->Norm1(value.data());
    EXPECT_NEAR(40, value[0], tolerance);
    EXPECT_NEAR(40, value[1], tolerance);
}

TEST(EpetraMultiVector, add)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);

    bool initialize_ensemble = true;
    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(map, global_num_vectors, initialize_ensemble);
    X.fill(1);
    EXPECT_EQ(global_num_vectors, X.getNumCols());
    EXPECT_EQ(global_num_elements, X.getNumRows());

    trrom::EpetraMultiVector Y(map, global_num_vectors, initialize_ensemble);
    Y.data()->operator()(0)->PutScalar(1);
    Y.data()->operator()(1)->PutScalar(2);
    Y.data()->operator()(2)->PutScalar(3);
    Y.data()->operator()(3)->PutScalar(4);

    // TEST 1
    X.update(1., Y, 1.);
    std::vector<double> value(global_num_vectors, -1.);
    X.data()->Norm1(value.data());
    double tolerance = 1e-6;
    EXPECT_NEAR(40, value[0], tolerance);
    EXPECT_NEAR(60, value[1], tolerance);
    EXPECT_NEAR(80, value[2], tolerance);
    EXPECT_NEAR(100, value[3], tolerance);

    // TEST 2
    X.update(2., Y, 1.);
    value.assign(value.size(), -1.);
    X.data()->Norm1(value.data());
    EXPECT_NEAR(80, value[0], tolerance);
    EXPECT_NEAR(140, value[1], tolerance);
    EXPECT_NEAR(200, value[2], tolerance);
    EXPECT_NEAR(260, value[3], tolerance);

    // TEST 3
    X.update(2., Y, 2.);
    value.assign(value.size(), -1.);
    X.data()->Norm1(value.data());
    EXPECT_NEAR(200, value[0], tolerance);
    EXPECT_NEAR(360, value[1], tolerance);
    EXPECT_NEAR(520, value[2], tolerance);
    EXPECT_NEAR(680, value[3], tolerance);
}

TEST(EpetraMultiVector, gemv_NT)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap map(global_num_elements, element_size, index_base, comm);

    bool initialize_ensemble = true;
    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(map, global_num_vectors, initialize_ensemble);
    X.data()->operator ()(0)->PutScalar(1);
    X.data()->operator ()(1)->PutScalar(2);
    X.data()->operator ()(2)->PutScalar(3);
    X.data()->operator ()(3)->PutScalar(4);

    Epetra_LocalMap input_map(global_num_vectors, index_base, comm);
    trrom::EpetraVector input(input_map);
    input.fill(1);

    // TEST 1: NOT TRANSPOSED
    trrom::EpetraVector output(map);
    X.gemv(false, 1., input, 0., output);

    trrom::EpetraVector gold(map);
    gold.fill(10);
    checkResults(gold, output);
}

TEST(EpetraMultiVector, gemv_T)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap parallel_map(global_num_elements, element_size, index_base, comm);

    bool initialize_ensemble = true;
    const int global_num_vectors = 4;
    trrom::EpetraMultiVector X(parallel_map, global_num_vectors, initialize_ensemble);
    X.data()->operator ()(0)->PutScalar(1);
    X.data()->operator ()(1)->PutScalar(2);
    X.data()->operator ()(2)->PutScalar(3);
    X.data()->operator ()(3)->PutScalar(4);

    trrom::EpetraVector input(parallel_map);
    input.fill(1);

    // TEST 1: NOT TRANSPOSED
    Epetra_LocalMap local_map(global_num_vectors, index_base, comm);
    trrom::EpetraVector output(local_map);
    X.gemv(true, 1., input, 0., output);

    trrom::EpetraVector gold(local_map);
    gold.operator [](0) = 20;
    gold.operator [](1) = 40;
    gold.operator [](2) = 60;
    gold.operator [](3) = 80;
    checkResults(gold, output);
}

TEST(DISABLED_EpetraMultiVector, gemm)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap parallel_map(global_num_elements, element_size, index_base, comm);

    bool initialize_ensemble = true;
    const int global_num_vectors = 4;
    trrom::EpetraMultiVector A(parallel_map, global_num_vectors, initialize_ensemble);
    trrom::EpetraMultiVector B(parallel_map, global_num_vectors, initialize_ensemble);
    A.data()->operator ()(0)->PutScalar(1);
    A.data()->operator ()(1)->PutScalar(2);
    A.data()->operator ()(2)->PutScalar(3);
    A.data()->operator ()(3)->PutScalar(4);
    B.update(1., A, 0.);

    // TEST 1: output = A * B'
    const int global_output_num_vectors = 20;
    trrom::EpetraMultiVector output(parallel_map, global_output_num_vectors, initialize_ensemble);
    A.gemm(false, true, 1., B, 0., output);

/*    trrom::EpetraMultiVector gold(parallel_map, global_output_num_vectors, initialize_ensemble);
    gold.fill(30);
    checkResults(gold, output);*/
}

TEST(EpetraMultiVector, getVector2)
{
    const int index_base = 0;
    const int element_size = 1;
    const int global_num_elements = 20;
    Epetra_MpiComm comm(MPI_COMM_WORLD);
    Epetra_BlockMap parallel_map(global_num_elements, element_size, index_base, comm);

    bool initialize_ensemble = true;
    const int global_num_vectors = 4;
    trrom::EpetraMultiVector A(parallel_map, global_num_vectors, initialize_ensemble);
    A.data()->operator ()(0)->PutScalar(1);
    A.data()->operator ()(1)->PutScalar(2);
    A.data()->operator ()(2)->PutScalar(3);
    A.data()->operator ()(3)->PutScalar(4);

    double gold = 20;
    double tolerance = 1e-6;
    EXPECT_NEAR(gold, A.getVector(0)->sum(), tolerance);
    gold = 40;
    EXPECT_NEAR(gold, A.getVector(1)->sum(), tolerance);
    gold = 60;
    EXPECT_NEAR(gold, A.getVector(2)->sum(), tolerance);
    gold = 80;
    EXPECT_NEAR(gold, A.getVector(3)->sum(), tolerance);

    // Repeat process again to determine if the Epetra_DataAccess::View mode is working as expected
    gold = 20;
    EXPECT_NEAR(gold, A.getVector(0)->sum(), tolerance);
    gold = 40;
    EXPECT_NEAR(gold, A.getVector(1)->sum(), tolerance);
    gold = 60;
    EXPECT_NEAR(gold, A.getVector(2)->sum(), tolerance);
    gold = 80;
    EXPECT_NEAR(gold, A.getVector(3)->sum(), tolerance);
}

}
