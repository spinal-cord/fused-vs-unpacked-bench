# Makefile for Fused vs Unpacked Ternary Benchmark
# Copyright (C) 2024 HyperFold Technologies UK Ltd.
# Multi-architecture support with SIMD optimizations

# Default compiler (can be overridden)
CC = gcc
CFLAGS = -O3 -Wall -Wextra
LDFLAGS =

# Detect OS and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# OS-specific adjustments
ifeq ($(UNAME_S), Darwin)
    # macOS specific settings
    CFLAGS += -mmacosx-version-min=10.12
    # On macOS, prefer clang for better Apple Silicon support
    CC = clang
    $(info Building for macOS)
    
    # Check if we're on Apple Silicon
    ifeq ($(UNAME_M), arm64)
        $(info Architecture: Apple Silicon (arm64))
        # On Apple Silicon, we can use NEON SIMD
    else ifeq ($(UNAME_M), arm64e)
        $(info Architecture: Apple Silicon (arm64e))
        # On Apple Silicon, we can use NEON SIMD
    else ifeq ($(UNAME_M), x86_64)
        $(info Architecture: Intel Mac (x86_64))
        # On Intel Mac, we can use AVX2
        CFLAGS += -march=native
    else
        $(warning Unknown architecture $(UNAME_M) on macOS)
    endif
    
else ifeq ($(UNAME_S), Linux)
    # Linux specific settings
    $(info Building for Linux)
    CFLAGS += -march=native
    LDFLAGS += -lrt
    
    ifeq ($(UNAME_M), x86_64)
        $(info Architecture: x86_64)
    else ifeq ($(UNAME_M), aarch64)
        $(info Architecture: ARM64 (aarch64))
        # On Linux ARM64, we can use NEON SIMD
    else
        $(warning Unknown architecture $(UNAME_M) on Linux)
    endif
    
else
    # Other Unix-like systems
    $(info Building for $(UNAME_S))
    LDFLAGS += -lrt
    ifeq ($(UNAME_M), x86_64)
        CFLAGS += -march=native
    endif
endif

TARGET = benchmark
SOURCE = benchmark.c

# Default target: scalar build (no SIMD)
all: $(TARGET)
	@echo "Build complete. Use 'make simd' for SIMD-optimized version."

# Build with SIMD support (architecture-specific)
simd:
ifeq ($(UNAME_S), Darwin)
    # macOS: Use clang with appropriate SIMD flags
    ifeq ($(UNAME_M), arm64)
        # Apple Silicon: NEON SIMD
		@echo "Building for Apple Silicon with NEON SIMD..."
		$(CC) $(CFLAGS) -DUSE_SIMD -o $(TARGET) $(SOURCE)
    else ifeq ($(UNAME_M), arm64e)
        # Apple Silicon: NEON SIMD
		@echo "Building for Apple Silicon with NEON SIMD..."
		$(CC) $(CFLAGS) -DUSE_SIMD -o $(TARGET) $(SOURCE)
    else ifeq ($(UNAME_M), x86_64)
        # Intel Mac: AVX2
		@echo "Building for Intel Mac with AVX2 SIMD..."
		$(CC) $(CFLAGS) -DUSE_SIMD -mavx2 -o $(TARGET) $(SOURCE)
    else
        # Unknown macOS architecture
		@echo "Unknown architecture $(UNAME_M). Building generic version."
		@$(MAKE) $(TARGET)
    endif
else ifeq ($(UNAME_S), Linux)
    # Linux: Use gcc with appropriate SIMD flags
    ifeq ($(UNAME_M), x86_64)
        # x86_64 Linux: AVX2
		@echo "Building for x86_64 Linux with AVX2 SIMD..."
		gcc $(CFLAGS) -DUSE_SIMD -mavx2 -o $(TARGET) $(SOURCE) $(LDFLAGS)
    else ifeq ($(UNAME_M), aarch64)
        # ARM64 Linux: NEON
		@echo "Building for ARM64 Linux with NEON SIMD..."
		gcc $(CFLAGS) -DUSE_SIMD -o $(TARGET) $(SOURCE) $(LDFLAGS)
    else
        # Unknown Linux architecture
		@echo "Unknown architecture $(UNAME_M). Building generic version."
		@$(MAKE) $(TARGET)
    endif
else
    # Other OS: Try generic SIMD build
	@echo "Building SIMD version for $(UNAME_S)..."
	$(CC) $(CFLAGS) -DUSE_SIMD -o $(TARGET) $(SOURCE) $(LDFLAGS)
endif

# Standard scalar build
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Build with SIMD and run
run-simd: simd
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Performance test: Build and run both scalar and SIMD versions
perf: clean
	@echo "=== Performance Comparison Test ==="
	@echo ""
	@echo "1. Building scalar version..."
	@$(MAKE) $(TARGET)
	@echo "Running scalar version..."
	@./$(TARGET) | tail -20
	@echo ""
	@echo "2. Building SIMD version..."
	@$(MAKE) simd
	@echo "Running SIMD version..."
	@./$(TARGET) | tail -20
	@echo ""
	@echo "=== Test Complete ==="

# Help
help:
	@echo "Fused vs Unpacked Ternary Computation Benchmark"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build scalar (non-SIMD) benchmark"
	@echo "  make simd     - Build with architecture-optimized SIMD"
	@echo "  make run      - Build scalar and run benchmark"
	@echo "  make run-simd - Build SIMD version and run"
	@echo "  make perf     - Build both versions and compare performance"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Architecture Support:"
	@echo "  - Apple Silicon (arm64/arm64e): Uses clang with NEON SIMD"
	@echo "  - Intel macOS (x86_64): Uses clang with AVX2 SIMD"
	@echo "  - x86_64 Linux: Uses gcc with AVX2 SIMD"
	@echo "  - ARM64 Linux: Uses gcc with NEON SIMD"

.PHONY: all simd run run-simd clean perf help