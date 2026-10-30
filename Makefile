NVCC ?= nvcc
ARCH ?= sm_89
TARGET := argon_gpu

all: $(TARGET)

$(TARGET): src/argon_gpu.cu
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
