# Makefile

###############################################################################
# 如果 CUDA 并非安装在 /lsc/opt/cuda-12.5，可自行修改此处
CUDA_PATH      ?= /lsc/opt/cuda-12.5

# 选择 NVCC 作为编译器（指定到具体 nvcc 路径）
NVCC           = $(CUDA_PATH)/bin/nvcc

# 编译选项 
# -std=c++14 或 -std=c++17 等，根据你代码需求
# -arch=sm_80 以及 gencode=arch=compute_80,code=sm_80 针对 GPU 架构 8.0 (Ampere)
# -O3 表示优化编译，如需要调试可换成 -g -G
NVCCFLAGS      = -std=c++14 -arch=sm_80 -gencode arch=compute_80,code=sm_80 -O3

# 包含目录（为了找到 cuda_runtime.h 等）
INC_PATH       = -I$(CUDA_PATH)/include

# 链接目录（如需链接特定 CUDA 库，可在这里添加 -lcudart 等）
LIB_PATH       = -L$(CUDA_PATH)/lib64

# 可执行文件名称
TARGET         = slic

# 源文件列表
SRCS           = main.cu data.cu

# 目标文件将由 .cu 转成 .o
OBJS           = $(SRCS:.cu=.o)

###############################################################################
# 默认规则：编译并生成可执行文件
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INC_PATH) $(LIB_PATH) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INC_PATH) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
