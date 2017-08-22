#Add source files here
EXECUTABLE	:= sobel5x5 
# Cuda source files (compiled with cudacc)
CUFILES		:= sobel5x5.cu 
CUDEPS		:= filter_kernel5x5.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 

################################################################################
# Rules and targets

include ../../common/common.mk
