# Add source files here
EXECUTABLE	:= Histogram64_gpu_atomic
# Cuda source files (compiled with cudacc)
CUFILES_sm_11	:= histogram64_gpu_atomic.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= #histogram_cpu.cpp


################################################################################
# Rules and targets

include ../../common/common.mk
