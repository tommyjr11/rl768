# Makefile

# 使用 NVCC 作为编译器
NVCC = nvcc

# 编译选项，可根据需求加上 -O3
NVCCFLAGS = -std=c++14 -arch=sm_80 -gencode=arch=compute_80,code=sm_80

# 可执行文件名
TARGET = slic

# 源文件列表
SRCS = main.cu data.cu constants.cu slic.cu help_function.cu

# 对应的目标文件
OBJS = $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
