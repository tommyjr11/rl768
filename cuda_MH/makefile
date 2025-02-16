# Makefile

# 使用 NVCC 作为编译器
NVCC = nvcc

# 编译选项，可按需调整。例如 -O3 进行优化
NVCCFLAGS = -std=c++14 -O3

# 目标可执行文件名
TARGET = slic

# 源文件列表
SRCS = main.cu data.cu

# 对应的目标文件
OBJS = $(SRCS:.cu=.o)

all: $(TARGET)

# 链接阶段：把所有 .o 链接生成可执行文件
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# 编译阶段：每个 .cu -> .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
