# Set SOL install directory
HOME=/Users/miguelaguilo/locus/trrom/

# Set MEX compiler (MEX) and corresponding flags (MEXFLAG)
MEXFLAG=-lblas
MEX=/Applications/MATLAB_R2015a.app/bin/mex 
MEXINC=/Applications/MATLAB_R2015a.app/extern/include/

# Set GNU compiler options
CXXOPTIMFLAGS=-O3
CXX=/usr/local/Cellar/gcc/4.9.2_1/bin/g++-4.9
CXXINCL=/usr/local/Cellar/gcc/4.9.2_1/include/c++/4.9.2/
CXXFLAGS=-I$(HOME)include -I$(CXXINCL) -I$(MEXINC) -g -fPIC -Wall -std=c++11 

# Set MEX source directory (MEXDIR), MEX include directory (MEXINC), objects output directory (OBJDIR), 
# and executables output directory (EXEDIR)
OBJDIR=obj/
EXEDIR=exe/


OBJS=$(HOME)matlab/TRROM_MxVector.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVectorTest.o \
	$(HOME)matlab/TRROM_MxMatrixTest.o \
	$(HOME)matlab/TRROM_MxDirectSolver.o \
	$(HOME)matlab/TRROM_MxDirectSolverTest.o \
	$(HOME)matlab/TRROM_MxSpectralDecompositionTest.o \
	$(HOME)matlab/TRROM_MxSingularValueDecomposition.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecomposition.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecompositionTest.o \
	$(HOME)matlab/TRROM_MxReducedObjectiveOperators.o \
	$(HOME)matlab/TRROM_MxReducedObjectiveOperatorsTest.o \
	$(HOME)matlab/TRROM_MxInequalityOperatorsTest.o
	
VECTEST=$(HOME)matlab/TRROM_MxVectorTest.o \
	$(HOME)matlab/TRROM_MxVector.o
	
MATTEST=$(HOME)matlab/TRROM_MxMatrixTest.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o

SVDTEST=$(HOME)matlab/TRROM_MxSpectralDecompositionTest.o \
	$(HOME)matlab/TRROM_MxSingularValueDecomposition.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o

	
OBJTEST=$(HOME)matlab/TRROM_MxReducedObjectiveOperatorsTest.o \
	$(HOME)matlab/TRROM_MxReducedObjectiveOperators.o \
	$(HOME)matlab/TRROM_MxVector.o

INQTEST=$(HOME)matlab/TRROM_MxInequalityOperatorsTest.o \
	$(HOME)matlab/TRROM_MxVector.o
	
ORTHOTEST=$(HOME)matlab/TRROM_MxOrthogonalDecompositionTest.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecomposition.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o
	
SOLVERTEST=$(HOME)matlab/TRROM_MxDirectSolverTest.o \
	$(HOME)matlab/TRROM_MxDirectSolver.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o

.C.o:
	$(MEX) -largeArrays CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) -outdir $(OBJDIR) -c $@ $<

test: $(OBJS) 

MX:
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(VECTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(MATTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(SVDTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(OBJTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(INQTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(ORTHOTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(SOLVERTEST)

all:
	make clean
	mkdir obj 
	mkdir exe
	make -f Makefile test
	make -f Makefile MX

clean:
	$(RM) -rf $(OBJDIR) $(EXEDIR)
	find . -name "*.o" -exec rm {} \;
