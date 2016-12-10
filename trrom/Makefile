# Set compiler options
CXX = /usr/local/bin/mpicxx
CXXFLAGS =-fopenmp -O3 -g -std=c++11
CXX_INCLUDE = /usr/local/Cellar/gcc/4.9.2_1/include/c++/4.9.2/

# Set source and include directories
SRC = /Users/miguelaguilo/locus/trrom/src/
INC = /Users/miguelaguilo/locus/trrom/include/

# Set source and include directories for unit tests
MOCK = /Users/miguelaguilo/locus/trrom/mock/
TEST = /Users/miguelaguilo/locus/trrom/unit_tests/

# Google test directory 
GTEST_INCLUDE_DIR = /Users/miguelaguilo/Software/gtest/include/
GTEST_SRC = /Users/miguelaguilo/Software/gtest_mybuild_clang++/CMakeFiles/gtest.dir/src/

# Set object files directory
OBJS = $(SRC)TRROM_DiscreteEmpiricalInterpolation.o \
	$(SRC)TRROM_KelleySachsAugmentedLagrangian.o \
	$(SRC)TRROM_AugmentedLagrangianDataMng.o \
	$(SRC)TRROM_ProjectedSteihaugTointPcg.o \
	$(SRC)TRROM_AugmentedLagrangianTypeLP.o \
	$(SRC)TRROM_AugmentedLagrangianTypeNP.o \
	$(SRC)TRROM_SpectralDecompositionMng.o \
	$(SRC)TRROM_ReducedSpaceAssemblyMng.o \
	$(SRC)TRROM_ReducedBasisAssemblyMng.o \
	$(SRC)TRROM_KelleySachsReducedBasis.o \
	$(SRC)TRROM_TrustRegionKelleySachs.o \
	$(SRC)TRROM_ReducedBasisInterface.o \
	$(SRC)TRROM_SteihaugTointNewtonIO.o \
	$(SRC)TRROM_SteihaugTointDataMng.o \
	$(SRC)TRROM_OptimizationDataMng.o \
	$(SRC)TRROM_SteihaugTointSolver.o \
	$(SRC)TRROM_ModifiedGramSchmidt.o \
	$(SRC)TRROM_ReducedBasisDataMng.o \
	$(SRC)TRROM_KelleySachsStepMng.o \
	$(SRC)TRROM_TrustRegionStepMng.o \
	$(SRC)TRROM_AssemblyMngTypeLP.o \
	$(SRC)TRROM_ReducedBasisData.o \
	$(SRC)TRROM_BoundConstraints.o \
	$(SRC)TRROM_SteihaugTointPcg.o \
	$(SRC)TRROM_BrandLowRankSVD.o \
	$(SRC)TRROM_ReducedHessian.o \
	$(SRC)TRROM_Preconditioner.o \
	$(SRC)TRROM_VariablesUtils.o \
	$(SRC)TRROM_ReducedBasis.o \
	$(SRC)TRROM_KelleySachs.o \
	$(SRC)TRROM_Rosenbrock.o \
	$(SRC)TRROM_Variable.o \
	$(SRC)TRROM_Control.o \
	$(SRC)TRROM_Circle.o \
	$(SRC)TRROM_Radius.o \
	$(SRC)TRROM_Slacks.o \
	$(SRC)TRROM_State.o \
	$(SRC)TRROM_Data.o \
	$(SRC)TRROM_Dual.o \
	TrromMain.o
	

UTEST = $(GTEST_SRC)gtest-all.cc.o \
	$(TEST)UnitTestMain.o \
	$(TEST)TRROM_AugmentedLagrangianTest.o \
	$(TEST)TRROM_SimpleTest.o
	
MOCKFILES =$(MOCK)SVD.o \
	$(MOCK)MatlabQR.o

.cpp.o:
	$(CXX) $(CXXFLAGS) -I$(CXX_INCLUDE) -I$(GTEST_INCLUDE_DIR) -I$(MOCK) -I$(INC) -I$(SRC) \
	-c -o $@ $<

trrom: $(OBJS)
	$(CXX) $(CXXFLAGS) -o trrom $^
	
utest: $(OBJS) $(UTEST) $(MOCKFILES)
	$(CXX) $(CXXFLAGS) -o utest $^

clean:
	find . -name "*.o" -exec rm {} \;
	rm -f trrom utest
