# Set compiler options
CXX = /usr/local/bin/mpicxx
CXXFLAGS = -Wall -Wuninitialized -Werror -Wformat -fopenmp -O3 -g -std=c++11
CXX_INCLUDE = /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include/c++/

# Set source and include directories
SRC = /Users/miguelaguilo/locus/xgen/src/
INC = /Users/miguelaguilo/locus/xgen/include/

# Set source and include directories for unit tests
TEST = /Users/miguelaguilo/locus/xgen/tests/

# Google test directory 
GTEST_INCLUDE = /Users/miguelaguilo/Software/gtest/include/
GTEST_BUILD = /Users/miguelaguilo/Software/gtest_mybuild_mpicxx/CMakeFiles/gtest.dir/src/

# Set object files directory
OBJS = $(SRC)

UTEST = $(GTEST_BUILD)gtest-all.cc.o \
	$(TEST)Locus_UnitTestMain.o \
	$(TEST)Locus_VectorTest.o

.cpp.o:
	$(CXX) $(CXXFLAGS) -I$(CXX_INCLUDE) -I$(GTEST_INCLUDE) -I$(INC) -I$(SRC) \
	-c -o $@ $<

locus: $(OBJS) LocusMain.o
	$(CXX) $(CXXFLAGS) -o locus $^
	
test: $(UTEST)
	$(CXX) $(CXXFLAGS) -o test $^

clean:
	find . -name "*.o" -exec rm {} \;
	rm -f locus test
