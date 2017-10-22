# Set compiler options
CXX = /usr/local/bin/mpicxx
CXXFLAGS = -Wall -Wuninitialized -Werror -Wformat -fopenmp -O3 -g -std=c++11
CXX_INCLUDE = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/

# Set source and include directories
LOCUS_SOURCE = /Users/miguelaguilo/locus/xgen/src/
LOCUS_INCLUDE = /Users/miguelaguilo/locus/xgen/include/

# Set specific include directories
LOCUS_OPERATORS_INCLUDE = /Users/miguelaguilo/locus/xgen/include/Operators/
LOCUS_CRITERION_INCLUDE = /Users/miguelaguilo/locus/xgen/include/Criterion/
LOCUS_DATA_INCLUDE = /Users/miguelaguilo/locus/xgen/include/DataStructures/
LOCUS_ALGEGRA_INCLUDE = /Users/miguelaguilo/locus/xgen/include/LinearAlgebra/
LOCUS_OC_INCLUDE = /Users/miguelaguilo/locus/xgen/include/OptimalityCriteria/
LOCUS_LINESEARCH_INCLUDE = /Users/miguelaguilo/locus/xgen/include/LineSearch/
LOCUS_TRUSTREGION_INCLUDE = /Users/miguelaguilo/locus/xgen/include/TrustRegion/
LOCUS_NLCG_INCLUDE = /Users/miguelaguilo/locus/xgen/include/NonlinearConjugateGradient/

# Set source and include directories for unit tests
UNIT_TEST = /Users/miguelaguilo/locus/xgen/tests/

# Google test directory 
GTEST_INCLUDE = /Users/miguelaguilo/Software/gtest/include/
GTEST_BUILD = /Users/miguelaguilo/Software/gtest_mybuild_mpicxx/CMakeFiles/gtest.dir/src/

# Set object files directory
OBJS = $(LOCUS_SOURCE)

UTEST = $(GTEST_BUILD)gtest-all.cc.o \
	$(UNIT_TEST)Locus_NonlinearConjugateGradientTest.o \
	$(UNIT_TEST)Locus_AugmentedLagrangianTest.o \
	$(UNIT_TEST)Locus_OptimalityCriteriaTest.o \
	$(UNIT_TEST)Locus_PythonInterfaceTest.o \
	$(UNIT_TEST)Locus_LinearAlgebraTest.o \
	$(UNIT_TEST)Locus_DataFactoryTest.o \
	$(UNIT_TEST)Locus_LineSearchTest.o \
	$(UNIT_TEST)Locus_UnitTestMain.o \
	$(UNIT_TEST)Locus_BoundsTest.o \
	$(UNIT_TEST)Locus_VectorTest.o

.cpp.o:
	$(CXX) $(CXXFLAGS) -I$(CXX_INCLUDE) -I$(GTEST_INCLUDE) -I$(UNIT_TEST) -I$(LOCUS_SOURCE) \
	-I$(LOCUS_INCLUDE) -I$(LOCUS_OC_INCLUDE) -I$(LOCUS_ALGEGRA_INCLUDE) -I$(LOCUS_OPERATORS_INCLUDE) \
	-I$(LOCUS_CRITERION_INCLUDE) -I$(LOCUS_TRUSTREGION_INCLUDE) -I$(LOCUS_NLCG_INCLUDE) \
	-I$(LOCUS_DATA_INCLUDE) -I$(LOCUS_LINESEARCH_INCLUDE) \
	-c -o $@ $<

locus: $(OBJS) LocusMain.o
	$(CXX) $(CXXFLAGS) -o locus $^
	
test: $(UTEST)
	$(CXX) $(CXXFLAGS) -o test $^

clean:
	find . -name "*.o" -exec rm {} \;
	rm -f locus test
