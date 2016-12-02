# Set SOL install directory
HOME=/scratch/maguilo/sierra/trrom/trrom/

# Set MEX compiler (MEX) and corresponding flags (MEX_FLAG)
MEX=/usr/local/matlab/8.6/bin/mex
MEX_FLAG=-cxx -lblas
MEXINC=/usr/local/matlab/8.6/extern/include/

# Set GNU compiler options
CXX=g++
CXXOPTIMFLAGS=O3

# Set MEX source directory (MEXDIR), MEX include directory (MEXINC), objects output directory (OBJDIR), 
# and executables output directory (EXEDIR)
OBJDIR=obj/
EXEDIR=exe/

CXXFLAGS=-I$(HOME)include -I$(MEXINC) -g -fPIC -Wall

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
	$(HOME)matlab/TRROM_MxReducedObjectiveOperatorsTest.o
	
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
	
ORTHOTEST=$(HOME)matlab/TRROM_MxOrthogonalDecompositionTest.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecomposition.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o
	
SOLVERTEST=$(HOME)matlab/TRROM_MxDirectSolverTest.o \
	$(HOME)matlab/TRROM_MxDirectSolver.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o

.C.o:
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(OBJDIR) $(CXXFLAGS) -c -o $@ $<

test: $(OBJS) 

MX:
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(VECTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(MATTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(SVDTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(OBJTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(ORTHOTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(SOLVERTEST)

all:
	make clean
	mkdir obj 
	mkdir exe
	make -f Makefile test
	make -f Makefile MX

clean:
	$(RM) -rf $(OBJDIR) $(EXEDIR)
	find . -name "*.o" -exec rm {} \;
