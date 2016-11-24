# Set SOL install directory
HOME=/scratch/maguilo/sierra/trrom/trrom/

# Set MEX compiler (MEX) and corresponding flags (MEX_FLAG)
MEX=/usr/local/matlab/8.2/bin/mex
MEX_FLAG=-cxx -lblas
MEXINC=/usr/local/matlab/8.6b/extern/include/

# Set GNU compiler options
CXX=g++
CXXOPTIMFLAGS=O3

# Set MEX source directory (MEXDIR), MEX include directory (MEXINC), objects output directory (OBJDIR), 
# and executables output directory (EXEDIR)
OBJDIR=obj/
EXEDIR=exe/

CXXFLAGS=-I$(HOME)include -I$(MEXINC) -g -fPIC

OBJS=$(HOME)matlab/TRROM_MxVector.o \
	$(HOME)matlab/TRROM_MxVectorTest.o \
	$(HOME)matlab/TRROM_MxMatrixTest.o
	
VECTEST=$(HOME)matlab/TRROM_MxVectorTest.o \
	$(HOME)matlab/TRROM_MxVector.o
	
	
MATTEST=$(HOME)matlab/TRROM_MxMatrixTest.o \
	$(HOME)matlab/TRROM_MxVector.o

.C.o:
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(OBJDIR) $(CXXFLAGS) -c -o $@ $<

test: $(OBJS) 

MX:
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(VECTEST)
	$(MEX) CXX=$(CXX) CXXOPTIMFLAGS=$(CXXOPTIMFLAGS) LD=$(CXX) $(MEX_FLAG) -outdir $(EXEDIR) $(MATTEST)

all:
	make clean
	mkdir obj 
	mkdir exe
	make -f Makefile test
	make -f Makefile MX

clean:
	$(RM) -rf $(OBJDIR) $(EXEDIR)
	find . -name "*.o" -exec rm {} \;
