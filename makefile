# Makefile

#############################
# 1. 基本设置
#############################

# 使用 NVCC 作为编译器
NVCC       := nvcc

# 可执行文件名
TARGET     := slic

# 源文件列表
SRCS       := main.cu data.cu

# 对应的目标文件 (把 .cu 后缀替换成 .o)
OBJS       := $(SRCS:.cu=.o)


#############################
# 2. 编译选项
#############################

# (a) 公共编译选项：
#    -std=c++14: 使用 C++14 标准
#    -arch=sm_80: 针对 NVIDIA Ampere 架构（如 A100，或 RTX 30 系列）
#    -gencode=arch=compute_80,code=sm_80: 再次指定生成 sm_80 的代码
COMMON_FLAGS := -std=c++14 -arch=sm_80 -gencode=arch=compute_80,code=sm_80

# (b) Release 模式下的优化选项
RELEASE_FLAGS := -O3 -Xptxas -O3

# (c) Debug 模式下的编译选项
#    -G: 生成用于调试的设备代码 (Device Debug)，会降低性能
#    -O0: 不做优化
DEBUG_FLAGS := -G -O0

# (d) 更严格的浮点精度选项(可选):
#    --fmad=false: 禁用 FMA 指令融合乘加（可减少某些舍入差异）
#    -prec-div=true / -prec-sqrt=true: 强化除法和开方的精度
# 一般建议仅在需要时启用，因为会影响性能
ACCURATE_FLOAT_FLAGS := --fmad=false -prec-div=true -prec-sqrt=true


#############################
# 3. 通过 make 参数控制编译模式
#############################
# 可以在命令行通过:
#   make          (默认使用 release 模式)
#   make DEBUG=1  (使用 debug 模式)
#   make ACCURATE=1 (启用准确浮点)
#
# 也可以组合，比如:
#   make DEBUG=1 ACCURATE=1
# 来同时打开调试+严格浮点(非常慢，但更易排查误差)
#############################

# 默认使用 release 模式
ifeq ($(DEBUG),1)
  NVCCFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS)
else
  NVCCFLAGS := $(COMMON_FLAGS) $(RELEASE_FLAGS)
endif

# 如果指定 ACCURATE=1，则追加严格浮点选项
ifeq ($(ACCURATE),1)
  NVCCFLAGS += $(ACCURATE_FLOAT_FLAGS)
endif


#############################
# 4. 规则部分
#############################

# 4.1 编译生成可执行文件
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# 4.2 编译 .cu -> .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 4.3 清理
clean:
	rm -f $(OBJS) $(TARGET)
