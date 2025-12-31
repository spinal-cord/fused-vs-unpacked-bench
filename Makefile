# Makefile for Fused vs Unpacked Ternary Benchmark
# Modified for cross-platform macOS support

CC = gcc
CFLAGS = -O3 -Wall -Wextra
LDFLAGS =

# Detect macOS and adjust flags
UNAME := $(shell uname -s)

ifeq ($(UNAME), Darwin)
    # macOS - no librt needed, don't use -march=native
    CFLAGS += -mmacosx-version-min=10.12
else ifeq ($(UNAME), Linux)
    # Linux - use librt and native optimizations
    CFLAGS += -march=native
    LDFLAGS += -lrt
else
    # Other Unix-like systems
    LDFLAGS += -lrt
endif

# Detect architecture
UNAME_M := $(shell uname -m)

# Disable -march=native and SIMD on Apple Silicon
ifeq ($(UNAME_M),arm64)
    CFLAGS += -DNO_SIMD
    $(info Building for Apple Silicon (arm64): SIMD disabled.)
else ifeq ($(UNAME_M),arm64e)
    CFLAGS += -DNO_SIMD
    $(info Building for Apple Silicon (arm64e): SIMD disabled.)
else ifeq ($(UNAME_M),x86_64)
    CFLAGS += -march=native
    $(info Building for Intel (x86_64): using -march=native.)
else
    $(warning Unknown architecture $(UNAME_M). Proceeding with generic build.)
endif

TARGET = benchmark
SOURCE = benchmark.c

# Default target
all: $(TARGET)

# Build with SIMD support (only for x86_64)
simd:
ifeq ($(UNAME_M),x86_64)
	@$(MAKE) CFLAGS="$(CFLAGS) -DUSE_SIMD -mavx2" $(TARGET)
else
	@echo "SIMD target is not supported on $(UNAME_M). Building generic version."
	@$(MAKE) $(TARGET)
endif

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Run with SIMD (if available)
run-simd: simd
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Help
help:
	@echo "Targets:"
	@echo "  make          - Build benchmark"
	@echo "  make simd     - Build with SIMD/AVX2 support (x86_64 only)"
	@echo "  make run      - Build and run benchmark"
	@echo "  make run-simd - Build with SIMD and run (x86_64 only)"
	@echo "  make clean    - Remove build artifacts"

.PHONY: all simd run run-simd clean help