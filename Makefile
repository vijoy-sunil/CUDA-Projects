# Add source files here
EXECUTABLE	:= vectorAdd
# Cuda source files (compiled with cudacc)
CUFILES		:= vectorAdd.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \

################################################################################
# Rules and targets

include ../../common/common.mk
