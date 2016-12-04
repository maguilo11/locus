#
# This makefile compiles the Intrepid / Matlab interface.
#
# This interface has dependencies on Trilinos packages.
#
# The file Makefile.export.Trilinos_install is used here
# since it will have all the dependencies required for
# a variety of required and optional Trilinos packages.
#
# Point to your own Makefile.export.Trilinos_install,
# and point to your own MEX compiler.
#
# Note that the Trilinos install should be a RELEASE install. 
#

# --> ACTION: include makefile export files ('install mode') for Trilinos; use RELEASE install
include /Users/miguelaguilo/Software/Trilinos/BUILD/release/Makefile.export.Trilinos_install

# --> ACTION: point to your mex compiler; use the mex option -largeArrayDims 
MEX = /Applications/MATLAB_R2015a.app/bin/mex

# Set compilers to Trilinos compilers
CXX = $(Trilinos_CXX_COMPILER)
CPP = $(Trilinos_C_COMPILER)

# Set flags to Trilinos flags
CPPFLAGS=$(Trilinos_C_COMPILER_FLAGS)
CXXFLAGS=-std=c++11
#CXXFLAGS = $(Trilinos_CXX_COMPILER_FLAGS)
CXXOPTIMFLAGS=-O3

# List sources for compilation
MOBJS = intrepid_setJacobian.mex64a \
        intrepid_setJacobianInv.mex64a \
        intrepid_setJacobianDet.mex64a \
        intrepid_getNumCubaturePoints.mex64a \
        intrepid_getCubature.mex64a \
        intrepid_getBasisValues.mex64a \
        intrepid_computeCellMeasure.mex64a \
        intrepid_HGRADtransformGRAD.mex64a \
        intrepid_HGRADtransformVALUE.mex64a \
        intrepid_multiplyMeasure.mex64a \
        intrepid_integrate.mex64a \
        intrepid_mapToPhysicalFrame.mex64a \
        intrepid_mapToReferenceFrame.mex64a \
        intrepid_scalarMultiplyDataField.mex64a \
        intrepid_scalarMultiplyDataData.mex64a \
        intrepid_dataIntegral.mex64a \
        intrepid_evaluate.mex64a \
        intrepid_multiplyVectorDataField.mex64a \
        intrepid_dotMultiplyDataField.mex64a \
        intrepid_vectorMultiplyDataField.mex64a \
        intrepid_dotMultiplyDataData.mex64a \
       	intrepid_computeEdgeMeasure.mex64a \
        intrepid_computeFaceMeasure.mex64a \
       	intrepid_getCellDeviatoricMat.mex64a \
       	intrepid_getCellVolumetricMat.mex64a \
       	intrepid_getPhysicalSideNormals.mex64a \
       	intrepid_mapToReferenceSubcell.mex64a \
       	intrepid_tensorMultiplyDataData.mex64a \
       	intrepid_tensorMultiplyDataField.mex64a \
		defGrad.mex64a \
		evaluateVectorField.mex64a \
		evaluateVectorGradField.mex64a \
		externalForceElastostaticsTL.mex64a \
		geometricBmat.mex64a \
		intrepid_det.mex64a \
		intrepid_inverse.mex64a \
		intrepid_transpose.mex64a \
		materialBmat.mex64a \
		strainEulerianLagrangian.mex64a \
		stress.mex64a \
		tangentStiffnessElastostaticsTL.mex64a \
		tensorMultiplyDataField.mex64a \
        mlmex.mex64a \
        myMatVecProductDataField.mex64a \
        myMatVecProductDataData.mex64a

default: mex

MEX: $(MOBJS)

%.o : %.cpp
	$(MEX) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) LD=$(CXX) CXXDEBUGFLAGS='' CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) -c $(Trilinos_INCLUDE_DIRS) $*.cpp

%.mex64a : %.cpp m2i_helpers.o
	$(MEX) CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) LD=$(CXX) CXXDEBUGFLAGS='' CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LDDEBUGFLAGS='' LDOPTIMFLAGS='-O3' \
	$(Trilinos_INCLUDE_DIRS) $(Trilinos_LIBRARY_DIRS) $(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES) $*.cpp m2i_helpers.o

clean:
	$(RM) $(OBJS) $(EXES) *.o *.mexa64 *.mexmaci64
