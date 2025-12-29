# Makefile for Fused vs Unpacked Ternary Benchmark
# Copyright (C) 2024 HyperFold Technologies UK Ltd.

CC = gcc
CFLAGS = -O3 -Wall -Wextra -march=native
LDFLAGS = -lrt

TARGET = benchmark
SOURCE = benchmark.c

# Default target
all: $(TARGET)

# Build with SIMD support
simd: CFLAGS += -DUSE_SIMD -mavx2
simd: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Run with SIMD
run-simd: simd
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Help
help:
	@echo "Fused vs Unpacked Ternary Computation Benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build benchmark"
	@echo "  make simd     - Build with SIMD/AVX2 support"
	@echo "  make run      - Build and run benchmark"
	@echo "  make run-simd - Build with SIMD and run"
	@echo "  make clean    - Remove build artifacts"

.PHONY: all simd run run-simd clean help
