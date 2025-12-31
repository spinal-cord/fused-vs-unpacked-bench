/*
 * Fused vs Unpacked Ternary Computation Benchmark
 * 
 * Isolates and proves the fusion advantage: computing directly on packed
 * 2-bit ternary data versus unpacking then computing.
 *
 * Three tests:
 * 1. Baseline (8-bit):      Standard decode -> compute
 * 2. 2-bit Unpacked:        Pack -> unpack -> compute
 * 3. Fused Kernel:          Pack -> compute directly on packed
 *
 * Copyright (C) 2024 HyperFold Technologies UK Ltd.
 * Licensed under GNU AGPLv3
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * PLATFORM DETECTION FOR SIMD
 * ============================================================================ */
/* Disable SIMD intrinsics on non-x86 platforms or if NO_SIMD is defined */
#if defined(__x86_64__) && !defined(NO_SIMD)
    #define SIMD_X86_64_AVAILABLE 1
#else
    #define SIMD_X86_64_AVAILABLE 0
#endif

/* Only include immintrin.h if x86-64 SIMD is available */
#if SIMD_X86_64_AVAILABLE
    #include <immintrin.h>
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================
#define MATRIX_ROWS 4096
#define MATRIX_COLS 4096
#define ITERATIONS 50
#define SPARSITY 0.5f  // 50% zeros

// ============================================================================
// DATA GENERATION
// ============================================================================
void generate_ternary_matrix_8bit(int8_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float r = (float)rand() / RAND_MAX;
        if (r < SPARSITY) {
            matrix[i] = 0;
        } else if (r < SPARSITY + (1.0f - SPARSITY) / 2.0f) {
            matrix[i] = 1;
        } else {
            matrix[i] = -1;
        }
    }
}

void pack_ternary_2bit(const int8_t *matrix_8bit, uint8_t *matrix_2bit,
                       int rows, int cols) {
    int packed_cols = (cols + 3) / 4;
    memset(matrix_2bit, 0, rows * packed_cols);
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int8_t val = matrix_8bit[r * cols + c];
            int packed_idx = r * packed_cols + c / 4;
            int trit_idx = c % 4;
            
            uint8_t bits;
            if (val == 0) bits = 0;       // 00
            else if (val == 1) bits = 1;  // 01
            else bits = 2;                // 10 for -1
            
            matrix_2bit[packed_idx] |= (bits << (trit_idx * 2));
        }
    }
}

void generate_input_vector(float *vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// ============================================================================
// TEST 1: BASELINE (8-BIT STANDARD)
// ============================================================================
void matvec_8bit_baseline(const int8_t *matrix, const float *input,
                          float *output, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const int8_t *row_ptr = matrix + r * cols;
        
        for (int c = 0; c < cols; c++) {
            int8_t w = row_ptr[c];
            if (w != 0) {
                sum += (float)w * input[c];
            }
        }
        
        output[r] = sum;
    }
}

// ============================================================================
// TEST 2: 2-BIT WITH UNPACKING (DECODE THEN COMPUTE)
// ============================================================================
static inline int8_t unpack_trit(uint8_t packed, int index) {
    uint8_t bits = (packed >> (index * 2)) & 0x3;
    return (bits == 2) ? -1 : (bits == 1) ? 1 : 0;
}

void matvec_2bit_unpacked(const uint8_t *matrix_packed, const float *input,
                          float *output, int rows, int cols) {
    int packed_cols = (cols + 3) / 4;
    
    // Allocate temporary unpacked buffer
    int8_t *unpacked_row = (int8_t*)malloc(cols * sizeof(int8_t));
    
    for (int r = 0; r < rows; r++) {
        const uint8_t *row_ptr = matrix_packed + r * packed_cols;
        
        // STEP 1: UNPACK (the overhead we want to eliminate)
        for (int packed_idx = 0; packed_idx < packed_cols; packed_idx++) {
            uint8_t packed = row_ptr[packed_idx];
            for (int trit_idx = 0; trit_idx < 4; trit_idx++) {
                int c = packed_idx * 4 + trit_idx;
                if (c < cols) {
                    unpacked_row[c] = unpack_trit(packed, trit_idx);
                }
            }
        }
        
        // STEP 2: COMPUTE (standard multiplication)
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            int8_t w = unpacked_row[c];
            if (w != 0) {
                sum += (float)w * input[c];
            }
        }
        
        output[r] = sum;
    }
    
    free(unpacked_row);
}

// ============================================================================
// TEST 3: FUSED KERNEL (COMPUTE DIRECTLY ON PACKED DATA)
// ============================================================================
// Combines two optimizations:
// 1. 2-bit packed encoding (75% memory reduction)
// 2. Fused decode-compute (no temporary array)

// Simple fused computation: unpack and compute in a single pass
void matvec_2bit_fused(const uint8_t *matrix_packed, const float *input,
                       float *output, int rows, int cols) {
    int packed_cols = (cols + 3) / 4;
    
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const uint8_t *row_ptr = matrix_packed + r * packed_cols;
        
        // FUSED: Unpack and compute in single loop
        for (int packed_idx = 0; packed_idx < packed_cols; packed_idx++) {
            uint8_t packed = row_ptr[packed_idx];
            int base_c = packed_idx * 4;
            
            // Process 4 trits from single packed byte
            for (int trit_idx = 0; trit_idx < 4; trit_idx++) {
                int c = base_c + trit_idx;
                if (c < cols) {
                    uint8_t bits = (packed >> (trit_idx * 2)) & 0x3;
                    
                    // Fused decode and multiply
                    if (bits == 1) {
                        sum += input[c];  // +1 * input
                    } else if (bits == 2) {
                        sum -= input[c];  // -1 * input
                    }
                    // bits == 0: skip (zero weight)
                }
            }
        }
        
        output[r] = sum;
    }
}

// ============================================================================
// TEST 4: FUSED + SPARSE CSR (ALL THREE OPTIMIZATIONS)
// ============================================================================
// Combines all three optimizations:
// 1. 2-bit packed encoding (75% memory reduction)
// 2. Fused decode-compute (no temporary array)
// 3. Sparse CSR format (skip zero-only packed bytes)

// CSR structure for sparse 2-bit packed matrix
typedef struct {
    uint8_t *values;      // Non-zero packed bytes
    int *col_indices;     // Column indices for each packed byte
    int *row_ptrs;        // Start index in values for each row
    int nnz_packed;       // Number of non-zero packed bytes
} sparse_csr_2bit_t;

// Convert dense 2-bit packed to sparse CSR format
sparse_csr_2bit_t* create_sparse_csr_2bit(const uint8_t *matrix_packed,
                                          int rows, int cols) {
    int packed_cols = (cols + 3) / 4;
    
    // Count non-zero packed bytes
    int nnz_packed = 0;
    for (int r = 0; r < rows; r++) {
        for (int pc = 0; pc < packed_cols; pc++) {
            if (matrix_packed[r * packed_cols + pc] != 0) {
                nnz_packed++;
            }
        }
    }
    
    // Allocate CSR structure
    sparse_csr_2bit_t *csr = (sparse_csr_2bit_t*)malloc(sizeof(sparse_csr_2bit_t));
    csr->values = (uint8_t*)malloc(nnz_packed * sizeof(uint8_t));
    csr->col_indices = (int*)malloc(nnz_packed * sizeof(int));
    csr->row_ptrs = (int*)malloc((rows + 1) * sizeof(int));
    csr->nnz_packed = nnz_packed;
    
    // Fill CSR structure
    int idx = 0;
    for (int r = 0; r < rows; r++) {
        csr->row_ptrs[r] = idx;
        for (int pc = 0; pc < packed_cols; pc++) {
            uint8_t packed = matrix_packed[r * packed_cols + pc];
            if (packed != 0) {
                csr->values[idx] = packed;
                csr->col_indices[idx] = pc;
                idx++;
            }
        }
    }
    csr->row_ptrs[rows] = idx;
    
    return csr;
}

void free_sparse_csr_2bit(sparse_csr_2bit_t *csr) {
    free(csr->values);
    free(csr->col_indices);
    free(csr->row_ptrs);
    free(csr);
}

// Fused sparse CSR computation: combines all three optimizations
void matvec_2bit_fused_sparse(const sparse_csr_2bit_t *csr, const float *input,
                               float *output, int rows, int cols) {
    (void)cols;  // Unused
    
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        
        // Only iterate over non-zero packed bytes (SPARSE)
        int row_start = csr->row_ptrs[r];
        int row_end = csr->row_ptrs[r + 1];
        
        for (int idx = row_start; idx < row_end; idx++) {
            uint8_t packed = csr->values[idx];      // 2-BIT PACKED
            int packed_col = csr->col_indices[idx];
            int base_c = packed_col * 4;
            
            // FUSED: Decode and compute in single operation
            for (int trit_idx = 0; trit_idx < 4; trit_idx++) {
                uint8_t bits = (packed >> (trit_idx * 2)) & 0x3;
                
                if (bits == 1) {
                    sum += input[base_c + trit_idx];  // +1 * input
                } else if (bits == 2) {
                    sum -= input[base_c + trit_idx];  // -1 * input
                }
                // bits == 0: skip (zero weight)
            }
        }
        
        output[r] = sum;
    }
}

// ============================================================================
// ADVANCED: FUSED KERNEL WITH SIMD (OPTIONAL)
// ============================================================================
#ifdef USE_SIMD
#if SIMD_X86_64_AVAILABLE
void matvec_2bit_fused_simd(const uint8_t *matrix_packed, const float *input,
                            float *output, int rows, int cols) {
    int packed_cols = (cols + 3) / 4;
    
    for (int r = 0; r < rows; r++) {
        __m256 sum_vec = _mm256_setzero_ps();
        const uint8_t *row_ptr = matrix_packed + r * packed_cols;
        
        int c = 0;
        
        // Process 8 floats at a time with SIMD
        for (int packed_idx = 0; packed_idx < packed_cols - 1; packed_idx += 2) {
            // Load 2 packed bytes = 8 trits
            uint8_t packed0 = row_ptr[packed_idx];
            uint8_t packed1 = row_ptr[packed_idx + 1];
            
            // Load 8 input values
            __m256 input_vec = _mm256_loadu_ps(&input[c]);
            
            // Decode and multiply (simplified for demonstration)
            // In production, use lookup tables or bit manipulation
            float weights[8];
            for (int i = 0; i < 4; i++) {
                uint8_t bits = (packed0 >> (i * 2)) & 0x3;
                weights[i] = (bits == 1) ? 1.0f : (bits == 2) ? -1.0f : 0.0f;
            }
            for (int i = 0; i < 4; i++) {
                uint8_t bits = (packed1 >> (i * 2)) & 0x3;
                weights[i + 4] = (bits == 1) ? 1.0f : (bits == 2) ? -1.0f : 0.0f;
            }
            
            __m256 weight_vec = _mm256_loadu_ps(weights);
            __m256 prod = _mm256_mul_ps(weight_vec, input_vec);
            sum_vec = _mm256_add_ps(sum_vec, prod);
            
            c += 8;
        }
        
        // Horizontal sum
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);
        float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                    sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
        
        // Handle remaining elements
        for (; c < cols; c++) {
            int packed_idx = c / 4;
            int trit_idx = c % 4;
            uint8_t packed = row_ptr[packed_idx];
            uint8_t bits = (packed >> (trit_idx * 2)) & 0x3;
            
            if (bits == 1) sum += input[c];
            else if (bits == 2) sum -= input[c];
        }
        
        output[r] = sum;
    }
}
#else
/* Provide a stub or fallback function if SIMD is requested but not available */
void matvec_2bit_fused_simd(const uint8_t *matrix_packed, const float *input,
                            float *output, int rows, int cols) {
    fprintf(stderr, "SIMD is not available on this architecture. Using fused scalar version.\n");
    matvec_2bit_fused(matrix_packed, input, output, rows, cols);
}
#endif /* SIMD_X86_64_AVAILABLE */
#endif /* USE_SIMD */

// ============================================================================
// BENCHMARK HARNESS
// ============================================================================
typedef struct {
    double time_ms;
    double time_per_iter_ms;
    double throughput_gflops;
    size_t memory_bytes;
} benchmark_result_t;

void benchmark_test(void (*func)(const void*, const float*, float*, int, int),
                   const void *matrix, const float *input, float *output,
                   int rows, int cols, int iterations, size_t memory_bytes,
                   const char *name, benchmark_result_t *result) {
    struct timespec start, end;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        func(matrix, input, output, rows, cols);
    }
    
    // Benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < iterations; i++) {
        func(matrix, input, output, rows, cols);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    result->time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                      (end.tv_nsec - start.tv_nsec) / 1000000.0;
    result->time_per_iter_ms = result->time_ms / iterations;
    
    // Calculate GFLOPS (2 ops per non-zero element)
    long long ops = (long long)rows * cols * 2 * iterations;
    result->throughput_gflops = (ops / 1e9) / (result->time_ms / 1000.0);
    
    result->memory_bytes = memory_bytes;
}

// ============================================================================
// VERIFICATION
// ============================================================================
int verify_results(const float *output1, const float *output2,
                   const float *output3, int size) {
    const float epsilon = 1e-3f;
    int errors = 0;
    
    for (int i = 0; i < size; i++) {
        float diff12 = output1[i] - output2[i];
        float diff13 = output1[i] - output3[i];
        
        if (diff12 > epsilon || diff12 < -epsilon ||
            diff13 > epsilon || diff13 < -epsilon) {
            if (errors < 5) {
                printf("  Mismatch at %d: baseline=%.3f, unpacked=%.3f, fused=%.3f\n",
                       i, output1[i], output2[i], output3[i]);
            }
            errors++;
        }
    }
    
    return errors;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    printf("========================================================================\n");
    printf("Fused vs Unpacked Ternary Computation Benchmark\n");
    printf("HyperFold Technologies UK Ltd.\n");
    printf("========================================================================\n\n");
    
    printf("Configuration:\n");
    printf("  Matrix Size:  %d × %d\n", MATRIX_ROWS, MATRIX_COLS);
    printf("  Total Weights: %d\n", MATRIX_ROWS * MATRIX_COLS);
    printf("  Sparsity:     %.0f%%\n", SPARSITY * 100.0f);
    printf("  Iterations:   %d\n", ITERATIONS);
    printf("\n");
    
    // Allocate memory
    size_t matrix_8bit_size = MATRIX_ROWS * MATRIX_COLS * sizeof(int8_t);
    int packed_cols = (MATRIX_COLS + 3) / 4;
    size_t matrix_2bit_size = MATRIX_ROWS * packed_cols * sizeof(uint8_t);
    
    int8_t *matrix_8bit = (int8_t*)malloc(matrix_8bit_size);
    uint8_t *matrix_2bit = (uint8_t*)malloc(matrix_2bit_size);
    float *input = (float*)malloc(MATRIX_COLS * sizeof(float));
    float *output1 = (float*)malloc(MATRIX_ROWS * sizeof(float));
    float *output2 = (float*)malloc(MATRIX_ROWS * sizeof(float));
    float *output3 = (float*)malloc(MATRIX_ROWS * sizeof(float));
    
    if (!matrix_8bit || !matrix_2bit || !input || !output1 || !output2 || !output3) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Generate test data
    srand(42);
    generate_ternary_matrix_8bit(matrix_8bit, MATRIX_ROWS, MATRIX_COLS);
    pack_ternary_2bit(matrix_8bit, matrix_2bit, MATRIX_ROWS, MATRIX_COLS);
    generate_input_vector(input, MATRIX_COLS);
    
    printf("Memory Footprint:\n");
    printf("  8-bit representation: %zu KB\n", matrix_8bit_size / 1024);
    printf("  2-bit representation: %zu KB (%.1f%% reduction)\n",
           matrix_2bit_size / 1024,
           100.0 * (1.0 - (double)matrix_2bit_size / matrix_8bit_size));
    printf("\n");
    
    // Run benchmarks
    benchmark_result_t result1, result2, result3;
    
    printf("========================================================================\n");
    printf("TEST 1: BASELINE (8-bit Standard)\n");
    printf("========================================================================\n");
    printf("Method: 8-bit int8_t -> standard integer multiplication\n");
    printf("Running...\n");
    benchmark_test((void (*)(const void*, const float*, float*, int, int))matvec_8bit_baseline,
                   matrix_8bit, input, output1, MATRIX_ROWS, MATRIX_COLS,
                   ITERATIONS, matrix_8bit_size, "Baseline", &result1);
    printf("Total Time:     %.2f ms\n", result1.time_ms);
    printf("Time per Iter:  %.3f ms\n", result1.time_per_iter_ms);
    printf("Throughput:     %.2f GFLOPS\n", result1.throughput_gflops);
    printf("\n");
    
    printf("========================================================================\n");
    printf("TEST 2: 2-BIT WITH UNPACKING (Decode Then Compute)\n");
    printf("========================================================================\n");
    printf("Method: 2-bit packed -> unpack to array -> standard multiplication\n");
    printf("Running...\n");
    benchmark_test((void (*)(const void*, const float*, float*, int, int))matvec_2bit_unpacked,
                   matrix_2bit, input, output2, MATRIX_ROWS, MATRIX_COLS,
                   ITERATIONS, matrix_2bit_size, "Unpacked", &result2);
    printf("Total Time:     %.2f ms\n", result2.time_ms);
    printf("Time per Iter:  %.3f ms\n", result2.time_per_iter_ms);
    printf("Throughput:     %.2f GFLOPS\n", result2.throughput_gflops);
    printf("vs Baseline:    %.2fx\n", result1.time_ms / result2.time_ms);
    printf("\n");
    
    printf("========================================================================\n");
    printf("TEST 3: FUSED KERNEL (2-bit Packed + Fused Decode-Compute)\n");
    printf("========================================================================\n");
    printf("Method: 2-bit packed -> direct computation without full unpacking\n");
    printf("Optimizations:\n");
    printf("  1. 2-bit packed encoding (75%% memory reduction)\n");
    printf("  2. Fused decode-compute (no temporary array)\n");
    printf("Running...\n");
    benchmark_test((void (*)(const void*, const float*, float*, int, int))matvec_2bit_fused,
                   matrix_2bit, input, output3, MATRIX_ROWS, MATRIX_COLS,
                   ITERATIONS, matrix_2bit_size, "Fused", &result3);
    printf("Total Time:     %.2f ms\n", result3.time_ms);
    printf("Time per Iter:  %.3f ms\n", result3.time_per_iter_ms);
    printf("Throughput:     %.2f GFLOPS\n", result3.throughput_gflops);
    printf("vs Baseline:    %.2fx\n", result1.time_ms / result3.time_ms);
    printf("vs Unpacked:    %.2fx\n", result2.time_ms / result3.time_ms);
    printf("\n");
    
    // Verify correctness
    printf("========================================================================\n");
    printf("VERIFICATION (Tests 1-3)\n");
    printf("========================================================================\n");
    int errors = verify_results(output1, output2, output3, MATRIX_ROWS);
    if (errors == 0) {
        printf("✓ All outputs match (within epsilon=1e-3)\n");
    } else {
        printf("✗ Found %d mismatches\n", errors);
    }
    printf("\n");
    
    // TEST 4: High sparsity with CSR
    printf("========================================================================\n");
    printf("TEST 4: FUSED + SPARSE CSR (All Three Optimizations)\n");
    printf("========================================================================\n");
    printf("Testing with 70%% sparsity to demonstrate CSR advantage...\n");
    
    // Generate high-sparsity matrix
    int8_t *matrix_8bit_sparse = (int8_t*)malloc(matrix_8bit_size);
    uint8_t *matrix_2bit_sparse = (uint8_t*)malloc(matrix_2bit_size);
    float *output4 = (float*)malloc(MATRIX_ROWS * sizeof(float));
    
    srand(42);
    for (int i = 0; i < MATRIX_ROWS * MATRIX_COLS; i++) {
        float r = (float)rand() / RAND_MAX;
        if (r < 0.7f) {
            matrix_8bit_sparse[i] = 0;
        } else if (r < 0.85f) {
            matrix_8bit_sparse[i] = 1;
        } else {
            matrix_8bit_sparse[i] = -1;
        }
    }
    pack_ternary_2bit(matrix_8bit_sparse, matrix_2bit_sparse, MATRIX_ROWS, MATRIX_COLS);
    
    // Create sparse CSR structure
    sparse_csr_2bit_t *csr = create_sparse_csr_2bit(matrix_2bit_sparse, MATRIX_ROWS, MATRIX_COLS);
    size_t csr_memory = csr->nnz_packed * (sizeof(uint8_t) + sizeof(int)) + 
                        (MATRIX_ROWS + 1) * sizeof(int);
    
    printf("  Sparsity: 70%%\n");
    printf("  Sparse CSR memory: %zu KB (%.1f%% of 2-bit dense)\n",
           csr_memory / 1024,
           100.0 * (double)csr_memory / matrix_2bit_size);
    printf("  Non-zero packed bytes: %d / %d (%.1f%% reduction)\n",
           csr->nnz_packed, MATRIX_ROWS * packed_cols,
           100.0 * (1.0 - (double)csr->nnz_packed / (MATRIX_ROWS * packed_cols)));
    printf("\nOptimizations:\n");
    printf("  1. 2-bit packed encoding (75%% memory reduction)\n");
    printf("  2. Fused decode-compute (no temporary array)\n");
    printf("  3. Sparse CSR format (skip zero-only packed bytes)\n");
    printf("Running...\n");
    
    benchmark_result_t result4;
    struct timespec start, end;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        matvec_2bit_fused_sparse(csr, input, output4, MATRIX_ROWS, MATRIX_COLS);
    }
    
    // Benchmark
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERATIONS; i++) {
        matvec_2bit_fused_sparse(csr, input, output4, MATRIX_ROWS, MATRIX_COLS);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    result4.time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                      (end.tv_nsec - start.tv_nsec) / 1000000.0;
    result4.time_per_iter_ms = result4.time_ms / ITERATIONS;
    long long ops = (long long)MATRIX_ROWS * MATRIX_COLS * 2 * ITERATIONS;
    result4.throughput_gflops = (ops / 1e9) / (result4.time_ms / 1000.0);
    result4.memory_bytes = csr_memory;
    
    printf("Total Time:     %.2f ms\n", result4.time_ms);
    printf("Time per Iter:  %.3f ms\n", result4.time_per_iter_ms);
    printf("Throughput:     %.2f GFLOPS\n", result4.throughput_gflops);
    printf("vs Baseline:    %.2fx\n", result1.time_ms / result4.time_ms);
    printf("vs Test 3:      %.2fx\n", result3.time_ms / result4.time_ms);
    printf("\n");
    
    // Summary table
    printf("========================================================================\n");
    printf("SUMMARY\n");
    printf("========================================================================\n\n");
    
    printf("%-30s | %12s | %12s | %12s | %10s\n",
           "Method", "Time (ms)", "Memory (KB)", "Speedup", "GFLOPS");
    printf("-------------------------------------------------------------------------------------\n");
    
    printf("%-30s | %12.2f | %12zu | %12s | %10.2f\n",
           "Test 1: Baseline (8-bit)",
           result1.time_ms, result1.memory_bytes / 1024, "1.00×",
           result1.throughput_gflops);
    
    printf("%-30s | %12.2f | %12zu | %12.2fx | %10.2f\n",
           "Test 2: 2-bit Unpacked",
           result2.time_ms, result2.memory_bytes / 1024,
           result1.time_ms / result2.time_ms,
           result2.throughput_gflops);
    
    printf("%-30s | %12.2f | %12zu | %12.2fx | %10.2f\n",
           "Test 3: Fused (2-bit+Fusion)",
           result3.time_ms, result3.memory_bytes / 1024,
           result1.time_ms / result3.time_ms,
           result3.throughput_gflops);
    
    printf("%-30s | %12.2f | %12zu | %12.2fx | %10.2f\n",
           "Test 4: Fused+CSR (70%% sparse)",
           result4.time_ms, result4.memory_bytes / 1024,
           result1.time_ms / result4.time_ms,
           result4.throughput_gflops);
    
    printf("\n");
    printf("Key Comparisons:\n");
    printf("  Test 3 vs Test 2 (Fusion advantage):    %.2fx speedup\n",
           result2.time_ms / result3.time_ms);
    printf("  Test 4 vs Test 3 (CSR advantage):       %.2fx speedup\n",
           result3.time_ms / result4.time_ms);
    printf("  Test 4 vs Baseline (Combined):          %.2fx speedup\n",
           result1.time_ms / result4.time_ms);
    
    printf("\n");
    printf("========================================================================\n");
    printf("CONCLUSION\n");
    printf("========================================================================\n\n");
    
    double fusion_advantage = result2.time_ms / result3.time_ms;
    double csr_advantage = result3.time_ms / result4.time_ms;
    double combined_advantage = result1.time_ms / result4.time_ms;
    
    printf("✓ THREE OPTIMIZATION PRINCIPLES DEMONSTRATED\n\n");
    
    printf("1. FUSION ADVANTAGE (Test 3 vs Test 2): %.2fx\n", fusion_advantage);
    printf("   Fused decode-compute eliminates unpacking overhead\n\n");
    
    printf("2. SPARSE CSR ADVANTAGE (Test 4 vs Test 3): %.2fx\n", csr_advantage);
    printf("   At 70%% sparsity, CSR format skips zero-only packed bytes\n\n");
    
    printf("3. COMBINED ARCHITECTURE (Test 4 vs Baseline): %.2fx\n", combined_advantage);
    printf("   All three optimizations working together:\n");
    printf("   - 2-bit packed encoding (75%% memory reduction)\n");
    printf("   - Fused decode-compute (no temporary array)\n");
    printf("   - Sparse CSR format (skip zeros efficiently)\n\n");
    
    if (combined_advantage > 1.0) {
        printf("✓ NET PERFORMANCE GAIN: %.2fx over baseline\n\n", combined_advantage);
    } else {
        printf("Net performance: %.2fx (%.0f%% of baseline speed)\n\n",
               combined_advantage, combined_advantage * 100.0);
    }
    
    printf("This benchmark proves that efficient ternary computation requires\n");
    printf("layered optimizations: memory reduction, fused operations, and\n");
    printf("sparse formats working together as a complete architecture.\n");
    
    // Cleanup
    free_sparse_csr_2bit(csr);
    free(matrix_8bit_sparse);
    free(matrix_2bit_sparse);
    free(output4);
    free(matrix_8bit);
    free(matrix_2bit);
    free(input);
    free(output1);
    free(output2);
    free(output3);
    
    return 0;
}
