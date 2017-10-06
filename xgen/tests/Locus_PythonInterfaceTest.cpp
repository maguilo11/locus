/*
 * Locus_PythonInterfaceTest.cpp
 *
 *  Created on: Oct 6, 2017
 *      Author: Miguel A. Aguilo Valentin
 */

#include <memory>

#include "Locus_Vector.hpp"
#include "Locus_Criterion.hpp"
#include "Locus_MultiVector.hpp"

namespace locus
{

template<typename ScalarType, typename OrdinalType = size_t>
class PythonInterface : public locus::Criterion<ScalarType, OrdinalType>
{
public:
    PythonInterface()
    {
    }
    virtual ~PythonInterface()
    {
    }

    ScalarType value(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                     const locus::MultiVector<ScalarType, OrdinalType> & aControl)
    {
        return (0);
    }
    void gradient(const locus::MultiVector<ScalarType, OrdinalType> & aState,
                  const locus::MultiVector<ScalarType, OrdinalType> & aControl,
                  locus::MultiVector<ScalarType, OrdinalType> & aOutput)
    {
    }
    std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> create() const
    {
        std::shared_ptr<locus::Criterion<ScalarType, OrdinalType>> tOutput =
                std::make_shared<locus::PythonInterface<ScalarType, OrdinalType>>();
        return (tOutput);
    }

private:
    PythonInterface(const locus::PythonInterface<ScalarType, OrdinalType> & aRhs);
    locus::PythonInterface<ScalarType, OrdinalType> & operator=(const locus::PythonInterface<ScalarType, OrdinalType> & aRhs);
};

template<typename ScalarType, typename OrdinalType = size_t>
class PythonVector
{
public:
    PythonVector()
    {
    }
    virtual ~PythonVector()
    {
    }

    //! Scales a Vector by a ScalarType constant.
    void scale(const ScalarType & aInput)
    {
    }
    //! Entry-Wise product of two vectors.
    void entryWiseProduct(const locus::Vector<ScalarType, OrdinalType> & aInput)
    {
    }
    //! Update vector values with scaled values of A, this = beta*this + alpha*A.
    void update(const ScalarType & aAlpha,
                const locus::Vector<ScalarType, OrdinalType> & aInputVector,
                const ScalarType & aBeta)
    {
    }
    //! Computes the absolute value of each element in the container.
    void modulus()
    {
    }
    //! Returns the inner product of two vectors.
    ScalarType dot(const locus::Vector<ScalarType, OrdinalType> & aInputVector) const
    {
    }
    //! Assigns new contents to the Vector, replacing its current contents, and not modifying its size.
    void fill(const ScalarType & aValue)
    {
    }
    //! Returns the number of local elements in the Vector.
    OrdinalType size() const
    {
    }
    //! Creates an object of type locus::Vector
    std::shared_ptr<locus::Vector<ScalarType, OrdinalType>> create() const
    {
    }
    //! Operator overloads the square bracket operator
    ScalarType & operator [](const OrdinalType & aIndex)
    {
    }
    //! Operator overloads the square bracket operator
    const ScalarType & operator [](const OrdinalType & aIndex) const
    {
    }

private:
    PythonVector(const locus::PythonVector<ScalarType, OrdinalType> & aRhs);
    locus::PythonVector<ScalarType, OrdinalType> & operator=(const locus::PythonVector<ScalarType, OrdinalType> & aRhs);
};

}

namespace LocusPythonInterfaceTest
{

/* ******************************************************************* */
/* ********************** PHYTON INTERFACE TESTS ********************* */
/* ******************************************************************* */

/*TEST(LocusTest, PythonInterface)
 {
 // ********* ALLOCATE DATA FACTORY *********
 std::shared_ptr<locus::DataFactory<double>> tDataFactory =
 std::make_shared<locus::DataFactory<double>>();
 const size_t tNumControls = 1;
 tDataFactory->allocateControl(tNumControls);

 // ********* ALLOCATE TRUST REGION ALGORITHM DATA MANAGER *********
 std::shared_ptr<locus::TrustRegionAlgorithmDataMng<double>> tDataMng =
 std::make_shared<locus::TrustRegionAlgorithmDataMng<double>>(*tDataFactory);
 double tScalarValue = 0.25;
 tDataMng->setInitialGuess(tScalarValue);
 tScalarValue = -1;
 tDataMng->setControlLowerBounds(tScalarValue);
 tScalarValue = 1;
 tDataMng->setControlUpperBounds(tScalarValue);

 // ********* ALLOCATE OBJECTIVE AND CONSTRAINT CRITERIA *********
 locus::PythonInterface<double> tPythonInterface;

 // ********* AUGMENTED LAGRANGIAN STAGE MANAGER *********
 std::shared_ptr<locus::AugmentedLagrangianStageMng<double>> tStageMng =
 std::make_shared<locus::AugmentedLagrangianStageMng<double>>(*tDataFactory, tPythonInterface);

 // ********* SET FIRST AND SECOND ORDER DERIVATIVE COMPUTATION PROCEDURES *********
 locus::AnalyticalGradient<double> tObjectiveGradient(tPythonInterface);
 tStageMng->setObjectiveGradient(tObjectiveGradient);

 locus::AnalyticalHessian<double> tObjectiveHessian(tPythonInterface);
 tStageMng->setObjectiveHessian(tObjectiveHessian);

 // ********* ALLOCATE KELLEY-SACHS ALGORITHM *********
 locus::KelleySachsAugmentedLagrangian<double> tAlgorithm(tDataFactory, tDataMng, tStageMng);
 tAlgorithm.solve();
 }*/

}
