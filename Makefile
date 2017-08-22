# Add source files here
EXECUTABLE	:= vectorReduce_atomic
# Cuda source files (compiled with cudacc)
CUFILES_sm_11	:= vectorReduce_atomic.cu

#CUFILES_sm_11	:= simpleAtomicIntrinsics.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \

################################################################################
# Rules and targets

include ../../common/common.mk


