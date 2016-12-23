# Set TRROM install directory
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

OBJS=$(HOME)matlab/TRROM_MxDriver.o \
	$(HOME)matlab/TRROM_MxVector.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVectorTest.o \
	$(HOME)matlab/TRROM_MxMatrixTest.o \
	$(HOME)matlab/TRROM_MxDirectSolver.o \
	$(HOME)matlab/TRROM_MxDirectSolverTest.o \
	$(HOME)matlab/TRROM_MxSpectralDecompositionTest.o \
	$(HOME)matlab/TRROM_MxSingularValueDecomposition.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecomposition.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecompositionTest.o \
	$(HOME)matlab/TRROM_MxReducedBasisObjective.o \
	$(HOME)matlab/TRROM_MxReducedObjectiveOperators.o \
	$(HOME)matlab/TRROM_MxReducedObjectiveOperatorsTest.o \
	$(HOME)matlab/TRROM_MxInequalityOperators.o \
	$(HOME)matlab/TRROM_MxInequalityOperatorsTest.o \
	$(HOME)matlab/TRROM_MxReducedBasisPDE.o \
	$(HOME)matlab/TRROM_MxReducedBasisPDETest.o \
	$(HOME)matlab/TRROM_MxBrandMatrixFactory.o \
	$(HOME)matlab/TRROM_MxBrandMatrixFactoryTest.o \
	$(HOME)matlab/TRROM_MxBrandLowRankSVD.o \
	$(HOME)matlab/TRROM_MxLowRankSVDTest.o \
	$(HOME)matlab/TRROM_MxLinearAlgebraFactory.o \
	$(HOME)matlab/TRROM_MxLinearAlgebraFactoryTest.o \
	$(HOME)matlab/TRROM_MxDiscreteEmpiricalInterpolationTest.o \
	$(HOME)matlab/TRROM_MxParsingUtilities.o \
	$(HOME)matlab/TRROM_MxTrustRegionReducedOrderModelTypeB.o \
	$(HOME)src/TRROM_Data.o \
	$(HOME)src/TRROM_Dual.o \
	$(HOME)src/TRROM_State.o \
	$(HOME)src/TRROM_Slacks.o \
	$(HOME)src/TRROM_Control.o \
	$(HOME)src/TRROM_Variable.o \
	$(HOME)src/TRROM_BrandLowRankSVD.o \
	$(HOME)src/TRROM_ReducedBasisData.o \
	$(HOME)src/TRROM_DiscreteEmpiricalInterpolation.o
	
EQTEST=$(HOME)matlab/TRROM_MxReducedBasisPDETest.o \
	$(HOME)matlab/TRROM_MxReducedBasisPDE.o \
	$(HOME)matlab/TRROM_MxVector.o \
	$(HOME)src/TRROM_Data.o \
	$(HOME)src/TRROM_Dual.o \
	$(HOME)src/TRROM_State.o \
	$(HOME)src/TRROM_Slacks.o \
	$(HOME)src/TRROM_Control.o \
	$(HOME)src/TRROM_Variable.o \
	$(HOME)src/TRROM_ReducedBasisData.o
	
VECTEST=$(HOME)matlab/TRROM_MxVectorTest.o \
	$(HOME)matlab/TRROM_MxVector.o
	
MATTEST=$(HOME)matlab/TRROM_MxMatrixTest.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o

SVDTEST=$(HOME)matlab/TRROM_MxSpectralDecompositionTest.o \
	$(HOME)matlab/TRROM_MxSingularValueDecomposition.o \
	$(HOME)matlab/TRROM_MxLinearAlgebraFactory.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o
	
OBJTEST=$(HOME)matlab/TRROM_MxReducedObjectiveOperatorsTest.o \
	$(HOME)matlab/TRROM_MxReducedObjectiveOperators.o \
	$(HOME)matlab/TRROM_MxVector.o

INQTEST=$(HOME)matlab/TRROM_MxInequalityOperatorsTest.o \
	$(HOME)matlab/TRROM_MxInequalityOperators.o \
	$(HOME)matlab/TRROM_MxVector.o

DEIMTEST=$(HOME)matlab/TRROM_MxDiscreteEmpiricalInterpolationTest.o \
	$(HOME)matlab/TRROM_MxLinearAlgebraFactory.o \
	$(HOME)matlab/TRROM_MxDirectSolver.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o \
	$(HOME)src/TRROM_DiscreteEmpiricalInterpolation.o

ORTHOTEST=$(HOME)matlab/TRROM_MxOrthogonalDecompositionTest.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecomposition.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o
	
SOLVERTEST=$(HOME)matlab/TRROM_MxDirectSolverTest.o \
	$(HOME)matlab/TRROM_MxDirectSolver.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o
	
BRANDFACTEST=$(HOME)matlab/TRROM_MxBrandMatrixFactoryTest.o \
	$(HOME)matlab/TRROM_MxBrandMatrixFactory.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o
	
BRANDALGOEST=$(HOME)matlab/TRROM_MxLowRankSVDTest.o \
	$(HOME)matlab/TRROM_MxSingularValueDecomposition.o \
	$(HOME)matlab/TRROM_MxOrthogonalDecomposition.o \
	$(HOME)matlab/TRROM_MxLinearAlgebraFactory.o \
	$(HOME)matlab/TRROM_MxBrandMatrixFactory.o \
	$(HOME)matlab/TRROM_MxBrandLowRankSVD.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o \
	$(HOME)src/TRROM_BrandLowRankSVD.o
	
LINALGFACTTEST=$(HOME)matlab/TRROM_MxLinearAlgebraFactoryTest.o \
	$(HOME)matlab/TRROM_MxLinearAlgebraFactory.o \
	$(HOME)matlab/TRROM_MxMatrix.o \
	$(HOME)matlab/TRROM_MxVector.o

.C.o:
	$(MEX) -largeArrays CXX=$(CXX) CXXFLAGS=$(CXXFLAGS) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) -outdir $(OBJDIR) -c $@ $<

test: $(OBJS) 

MX:
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(EQTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(VECTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(MATTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(SVDTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(OBJTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(INQTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(DEIMTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(ORTHOTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(SOLVERTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(BRANDFACTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(BRANDALGOEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEXFLAG) -outdir $(EXEDIR) $(LINALGFACTTEST)

all:
	make clean
	mkdir obj 
	mkdir exe
	make -f Makefile test
	make -f Makefile MX

clean:
	$(RM) -rf $(OBJDIR) $(EXEDIR)
	find . -name "*.o" -exec rm {} \;
