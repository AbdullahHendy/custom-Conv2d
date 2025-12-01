#pragma once
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

namespace eecs471 {

    // Constants for WMMA kernel
    // Output Tile Sizes
    #define BM 32
    #define BK 16
    #define BN 64
    
    // Warps that will process these output tiles
    #define WARPS_PER_BLOCK 4
    #define WARP_SIZE 32

    // Chunk of the output tile each warp will compute
    #define WM 16
    #define WN 32
    
    // WMMA fragment sizes
    // Available: https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-shape
    #define WMMA_M 16
    #define WMMA_N 16
    #define WMMA_K 16

    // Padding to reduce shared memory bank conflicts
    #define PAD 8

    // This kernel uses WMMA to perform the implicit GEMM convolution and unrolls W and X on-the-fly.
    // This kernels works in heirarchical manner:
    // 1. Grid of Blocks: Each block computes a BM x BN tile of the output matrix Y
    // 2. Warps within Block, there are WARPS_PER_BLOCK warps: Each warp computes a WM x WN chunk of the bigger BM x BN tile
    // 3. WMMA Fragments within Warp: Each warp uses WMMA to compute the WM x WN chunk in smaller WMMA_M x WMMA_N fragments
    // In the case of B=10000, M=24, C=12, H=33, W=33, K=7, H_out=27, W_out=27, BM=32, BK=16, BN=64:
    // - Each block computes a 32 x 64 tile of the output matrix Y
    // - Each block has 4 warps, where each warp computes a 16 x 32 chunk of the 32 x 64 tile
    // - Each warp uses WMMA to compute the 16 x 32 chunk in 1 x 2 fragments of size 16 x 16
    // The kernel unrolls W from (M, C, K, K) to (M, C*K*K) and X from (B, C, H, W) to (C*K*K, B*H_out*W_out) on-the-fly.
    // See the detailed comments inside the kernel for more information.
    template <int B, int M, int C, int H, int W, int K, int H_out, int W_out>
    __global__ void implicitUnrollWmmaTC(
        const float* __restrict__ W_global, // Weight Tensor (M, C, K, K)
        const float* __restrict__ X_global, // Input Tensor  (B, C, H, W)
        float* __restrict__ Y_global)       // Output Tensor (B, M, H_out, W_out)
    {


        int tid = threadIdx.x;
        int warpId = tid / WARP_SIZE; // Which warp out of the WARPS_PER_BLOCK warps in this block
        // We are launching WARPS_PER_BLOCK * WARP_SIZE threads per block
        // They are divided into WARPS_PER_BLOCK/2 warps in M dimension and WARPS_PER_BLOCK/2 warps in N dimension
        // In teh case of 4 WARPS_PER_BLOCK, we set: 2 warps in M dimension and 2 warps in N dimension
        int warpRow = (warpId / (WARPS_PER_BLOCK / 2)) * WM;
        int warpCol = (warpId % (WARPS_PER_BLOCK / 2)) * WN;


        // The Global "Matrix" Dimensions
        const int GEMM_M = M;                 // 24
        const int GEMM_K = C * K * K;         // 12 * 7 * 7 = 588
        const int GEMM_N = B * H_out * W_out; // 10000 * 27 * 27

        // Block Offsets
        // We are launching ceil(GEMM_M / BM) x ceil(GEMM_N / BN) blocks
        // Each block computes a BM x BN tile of the output matrix Y
        // In the case of M=24, BM=32, we launch 1 block in M dimension that computes a 32-row tile (only first 24 rows are valid)
        // In the case of N=7290000, BN=64, we launch 113907 blocks in N dimension
        int blockRow = blockIdx.y * BM; // Really only 1 block in M dimension
        int blockCol = blockIdx.x * BN; // 0 .. 113906 in N dimension

        // 1. Shared Memory (Padded for Bank Conflicts)
        // We later convert Float -> Half during the load to SMEM since inputs are Floats
        // Using union to reduce shared memory usage since we only don't need W and X after compute and we dont need Y before compute
        // Transpose W and X in shared memory for better memory access patterns during WMMA load
        __shared__ union {
            struct {
                half sW[BK][BM + PAD];
                half sX[BN][BK + PAD];
            } in;
            struct {
                float sY[BM][BN + PAD];
            } out;
        } smem;
        // Helper pointers
        half (*smem_W)[BM + PAD] = smem.in.sW;
        half (*smem_X)[BK + PAD] = smem.in.sX;
        float (*smem_Y)[BN + PAD] = smem.out.sY;


        // 2. Setup Accumulators
        // Accumulators within each warp we have WM/WMMA_M in the M dimension and WN/WMMA_N in the N dimension.
        // In the case of WM=16, WN=32, WMMA_M=16, WMMA_N=16, each warp has 1 fragment in M dimension and 2 fragments in N dimension.
        const int num_m_frags = WM / WMMA_M; // 1
        const int num_n_frags = WN / WMMA_N; // 2
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[num_m_frags][num_n_frags];

        // Initialize the output fragments to zero
        for (int i = 0; i < num_m_frags; ++i)
            for (int j = 0; j < num_n_frags; ++j)
                wmma::fill_fragment(c_frag[i][j], 0.0f);

        // 3. Fragments for WMMA
        // Each warp computes WM/WMMA_M fragments in M dimension and WN/WMMA_N fragments in N dimension.
        // In the case of WM=16, WN=32, WMMA_M=16, WMMA_N=16, each warp has 1 fragment in M dimension and 2 fragments in N dimension.
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag[num_m_frags];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[num_n_frags];

        // 3. Main Loop over K (GEMM_K = 588)
        // We step through GEMM_K in chunks of BK i.e. the width of our TILE in K dimension
        for (int k_step = 0; k_step < GEMM_K; k_step += BK) {
            // a. Load W (Matrix A in GEMM terminology) into Shared Memory
            // Each block has WARPS_PER_BLOCK * WARP_SIZE threads per block
            // Each block loads BM * BK elements, meaning each thread loads (BM * BK) / (WARPS_PER_BLOCK * WARP_SIZE) elements
            // In the case of 4 WARPS_PER_BLOCK, each block has 4*32 = 128 threads
            // In the case of BM=32, BK=16, each block loads 32*16 = 512 elements
            // Therefore, each thread loads 512 / 128 = 4 elements
            // Vectorization:
            // Instead of each thread loading (BM * BK) / (WARPS_PER_BLOCK * WARP_SIZE) elements one by one, 
            // load 4 at a time using float4 for better memory throughput
            // In the above case of 4 elements per thread, each thread loads exactly 1 vector of float4
            // NOTE: We dont unroll W since it is "contiguous" in memory already (M, C, K, K) and we want W to be (M, C*K*K)
            const int VECTOR_SIZE = 4; // Number of floats in a float4
            const int total_vectors = (BM * BK) / VECTOR_SIZE; // Total number of float4 vectors to load for one W tile
            const int vectors_per_thread = total_vectors / (WARPS_PER_BLOCK * WARP_SIZE); // Number of float4 vectors each thread loads
            const int vectors_per_row = BK / VECTOR_SIZE; // Number of float4 vectors per row in the W tile

            const float4* W_vec_ptr = reinterpret_cast<const float4*>(W_global); // cast W_global to float4 pointer for vectorized loads

            // TODO: This loop doesn't generalize and might wanna change it later if we need to test other TILE sizes (BM, BK, BN)
            for (int i = 0; i < vectors_per_thread; i++) { // In the above case, this loop runs 1 time
                int vec_idx = tid + i * (WARPS_PER_BLOCK * WARP_SIZE); // Stride by total threads in block for better coalescing
                // Decode vec_idx into (r, c) within the tile since vec_idx goes from 0 to (BM*BK)/VECTOR_SIZE - 1
                // In the case of BM=32, BK=16, r in [0..31], c in [0..3] since there are 4 vectors per row
                int r = vec_idx / vectors_per_row;
                int vec_c = vec_idx % vectors_per_row;
                int c = vec_c * VECTOR_SIZE; // Convert vector column index to actual element column index

                int globalRow = blockRow + r;
                int globalCol = k_step + c;

                if (globalRow < M && globalCol < GEMM_K) {
                    float4 vec_val = W_vec_ptr[(globalRow * GEMM_K + globalCol) / VECTOR_SIZE]; // globalRow is M dim, globalCol is C*K*K dim

                    // Store the 4 elements into shared memory as half
                    smem_W[c + 0][r] = __float2half(vec_val.x);
                    smem_W[c + 1][r] = __float2half(vec_val.y);
                    smem_W[c + 2][r] = __float2half(vec_val.z);
                    smem_W[c + 3][r] = __float2half(vec_val.w);

                } else {
                    smem_W[c + 0][r] = __float2half(0.0f);
                    smem_W[c + 1][r] = __float2half(0.0f);
                    smem_W[c + 2][r] = __float2half(0.0f);
                    smem_W[c + 3][r] = __float2half(0.0f);
                }
            }

            // b. Load X (Matrix B in GEMM terminology) into Shared Memory
            // Each block has WARPS_PER_BLOCK * WARP_SIZE threads per block
            // Each block loads BK * BN elements, meaning each thread loads (BK * BN) / (WARPS_PER_BLOCK * WARP_SIZE) elements
            // In the case of 4 WARPS_PER_BLOCK, each block has 4*32 = 128 threads
            // In the case of BK=16, BN=64, each block loads 16*64 = 1024 elements
            // Therefore, each thread loads 1024 / 128 = 8 elements
            // NOTE: We are doing strided loads for coalescing. Instead of thread 0 loading elements 0,1,2,3, it loads 0,128,256,384
            // NOTE: We need to unroll X from (B, C, H, W) to (C*K*K, B*H_out*W_out) on-the-fly during the load into shared memory
            
            // NOTE: The following static assert is to ensure that WARRPS_PER_BLOCK * WARP_SIZE is a multiple of BN
            // Given that condition then c = idx % BN = tid % BN (idx = tid + i * WARPS_PER_BLOCK * WARP_SIZE) (tid doesn't depend on i but idx does)
            // This allows the compiler to take c calculation (modulus operation) and other variables dependent on it
            // out of the loop since tid is constant for a thread
            static_assert((WARPS_PER_BLOCK * WARP_SIZE) % BN == 0, "WARPS_PER_BLOCK * WARP_SIZE must be multiple of BN");
            
            const int elements_per_thread_X = (BK * BN) / (WARPS_PER_BLOCK * WARP_SIZE); // 8
            for (int i = 0; i < elements_per_thread_X; i++) {
                int idx = tid + i * WARPS_PER_BLOCK * WARP_SIZE;
                // Decode idx into (r, c) within the tile since idx goes from 0 to BK*BN-1
                // In the case of BK=16, BN=64, r in [0..15], c in [0..63]
                int r = idx / BN;
                int c = tid % BN; // idx % BN = tid % BN due to static_assert above

                int global_k = k_step + r;
                int global_n = blockCol + c;

                half val = __float2half(0.0f);

                // Unroll X from (B, C, H, W) to (C*K*K, B*H_out*W_out) on-the-fly during the load into shared memory
                if (global_k < GEMM_K && global_n < GEMM_N) {
                    // Decode the global_k to to get (c, p, q), which are the channel and filter indices
                    // The dimensions are nested as: c (slowest changing) --> p (medium) --> q (fastest changing).
                    // Below c is K*K dims, below p is K dims
                    int c = global_k / (K * K); // Get the channel index by dividing global_k by the dims under it (K*K)
                    int pq_idx = global_k % (K * K); // Get the index within the K*K block
                    int p = pq_idx / K; // Get the filter row index by dividing the index within K*K by the dims under it (K)
                    int q = pq_idx % K; // Get the filter column index by taking modulus K

                    // Decode the global_n to get (b, h_out, w_out), which are the batch index and output spatial indices
                    // The dimensions are nested as: b (slowest changing) --> h_out (medium) --> w_out (fastest changing).
                    // Below b is H_out*W_out dims, below h_out is W_out dims
                    const int HW_out = H_out * W_out;
                    int b = global_n / HW_out; // Get the batch index by dividing global_n by the dims under it (H_out*W_out)
                    int hw_out_idx = global_n % HW_out; // Get the index within the H_out*W_out block
                    int h_out   = hw_out_idx / W_out; // Get the output height index by dividing the index within H_out*W_out by the dims under it (W_out)
                    int w_out   = hw_out_idx % W_out; // Get the output width index by taking modulus W_out

                    // Calculate the corresponding input spatial indices to pull from x
                    // At this point, the indices b and c are already correct. 
                    // The output position (h_out, w_out) indicates where the top-left corner of the filter is positioned on the input image (see Lecture 16 slide 59). 
                    // The filter offset (p,q) tells us which element within that patch we are looking for.

                    int h_in = h_out + p;
                    int w_in = w_out + q;

                    // Load if valid
                    if (h_in < H && w_in < W) {
                        // X layout: [B, C, H, W]
                        int x_idx = b * (C * H * W) +
                                    c * (H * W) +
                                    h_in * W +
                                    w_in;
                        val = __float2half(X_global[x_idx]);
                    }
                }
                smem_X[c][r] = val; // Write transposed into shared memory
            }

            // Make sure the tiles are loaded before using them
            // NOTE: We use __syncthreads() instead of __syncwarp() because each block has multiple warps that need to sync
            __syncthreads();


            // c. Compute the WMMA Matrix Multiplication
            // Within each warp, compute the WM x WN output tile
            // We loop since the BK dimension may be larger than WMMA_K and wmma needs to be done in chunks of WMMA_K
            // In the case of BK=16 and WMMA_K=16, we have only 1 iteration
            for (int sub_k = 0; sub_k < BK; sub_k += WMMA_K) {
                // Each warp computes WM/WMMA_M fragments in M dimension and WN/WMMA_N fragments in N dimension.
                // In the case of WM=16, WN=32, WMMA_M=16, WMMA_N=16, each warp has 1 fragment in M dimension and 2 fragments in N dimension.
                // We calculated those values earlier as num_m_frags and num_n_frags

                // Load A fragments from shared memory (transposed)
                for (int i = 0; i < num_m_frags; ++i) {
                    half* ptrW = &smem_W[sub_k][warpRow + i * WMMA_M];
                    wmma::load_matrix_sync(a_frag[i], ptrW, BM + PAD); // Load into a_frag starting from ptrW and stride BM + PAD
                }

                // Load B fragments from shared memory (transposed)
                for (int j = 0; j < num_n_frags; ++j) {
                    half* ptrX = &smem_X[warpCol + j * WMMA_N][sub_k];
                    wmma::load_matrix_sync(b_frag[j], ptrX, BK + PAD); // Load into b_frag starting from ptrX and stride BK + PAD
                }

                // Matrix Multiply-Accumulate using WMMA
                for (int i = 0; i < num_m_frags; ++i) {
                    for (int j = 0; j < num_n_frags; ++j) {
                        // Perform the matrix multiplication and accumulate
                        wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }

            // Ensure all warps are done computing before loading new tiles into shared memory
            __syncthreads();
        }

        // 4. Store to Shared Memory
        // Each warp has WM/WMMA_M fragments in M dimension and WN/WMMA_N fragments in N dimension.
        // In the case of WM=16, WN=32, WMMA_M=16, WMMA_N=16, each warp has 1 fragment in M dimension and 2 fragments in N dimension.
        // We store the fragments back to shared memory first before writing to global memory
        for (int i = 0; i < num_m_frags; ++i) {
            for (int j = 0; j < num_n_frags; ++j) {
                float* ptrY = &smem_Y[warpRow + i * WMMA_M][warpCol + j * WMMA_N];
                wmma::store_matrix_sync(ptrY, c_frag[i][j], BN + PAD, wmma::mem_row_major);
            }
        }

        // Ensure all warps are done storing to shared memory before writing to global memory
        __syncthreads();

        // 5. Write to Global Memory
        // WARPS_PER_BLOCK * WARP_SIZE threads per block write the BM x BN tile from shared memory to global memory
        // In the case of 4 WARPS_PER_BLOCK, each block has 4*32 = 128 threads
        // In the case of BM=32, BN=64, each block has 32*64 = 2048 elements
        // Therefore, each thread writes 2048 / 128 = 16 elements
        const int elements_per_thread_Y = (BM * BN) / (WARPS_PER_BLOCK * WARP_SIZE); // 16
        for (int i = 0; i < elements_per_thread_Y; i++) {
            int idx = tid + i * (WARPS_PER_BLOCK * WARP_SIZE);
            // Decode idx into (r, c) within the tile since idx goes from 0 to BM*BN-1
            // In the case of BM=32, BN=64, r in [0..31], c in [0..63]
            int r = idx / BN;
            int c = idx % BN;

            int globalRow = blockRow + r;
            int globalCol = blockCol + c;

            if (globalRow < M && globalCol < GEMM_N) {
                // Similar to logic used when loading X to decode globalCol back to (b, h_out, w_out)
                const int HW_out = H_out * W_out;
                int b = globalCol / HW_out;
                int hw_out_idx = globalCol % HW_out;
                int h_out = hw_out_idx / W_out;
                int w_out = hw_out_idx % W_out;

                int y_idx = (b) * (M * H_out * W_out) +
                            (globalRow) * (H_out * W_out) + // globalRow is m
                            (h_out) * (W_out) +
                            w_out;
                Y_global[y_idx] = smem_Y[r][c];
            }
        }
    }

    // TILE_H = Height of the output tile
    // TILE_W = Width of the output tile
    template<int B, int C, int M, int H, int W, int K, int H_out, int W_out, int TILE_H, int TILE_W>
    __global__ void convTiled(
        const float* __restrict__ x,
        const float* __restrict__ w,
        float* __restrict__ y)
    {
        // Define the shared memory arrays for Input Tile and Filters Tile
        // Load the whole weight tensor into shared memory since we will do the convolution for ALL M filters on the same input tile
        __shared__ float sW[M][C][K][K];
        __shared__ float sX[C][TILE_H + K - 1][TILE_W + K - 1]; // Input tile with halo for convolution

        // 0. Calculate Global Indices
        // The compiler knows TILE_H, TILE_W are constants.
        const int b = blockIdx.z;  // batch index

        // Base output coordinates for this block
        const int h_out_base = blockIdx.y * TILE_H;
        const int w_out_base = blockIdx.x * TILE_W;

        const int th = threadIdx.y;
        const int tw = threadIdx.x;

        // Global output coords for this thread
        const int h_out = h_out_base + th;
        const int w_out = w_out_base + tw;

        // 1. Load Filters into Shared Memory (sW)
        
        // Linearize shared memory to avoid complex division/modulus inside the loading loop
        // Load 4 floats at a time using float4 for better memory throughput
        const float4* w_vec = reinterpret_cast<const float4*>(w);
        float* sW_linear = &sW[0][0][0][0]; 
        float4* sW_vec = reinterpret_cast<float4*>(sW_linear);

        const int total_w_elements = M * C * K * K; // Total elements in weight tensor
        const int threads_per_block = blockDim.x * blockDim.y; // TILE_H * TILE_W
        int lin_tid = th * blockDim.x + tw; // Flattened thread index within block

        // Copy weights into shared memory
        for (int i = lin_tid; i < total_w_elements / 4; i += threads_per_block) {
             sW_vec[i] = w_vec[i];
        }

        // 2. Load Input Tile into Shared Memory (sX)
        // We need to load the input tile for all channels C.
        // The input region (TILE_H + K - 1) is larger than the output tile (TILE_H) and therefore the two inner loops are needed.
        
        const int tile_h = TILE_H + K - 1;
        const int tile_w = TILE_W + K - 1;

        for (int c = 0; c < C; ++c) {
            for (int yy = th; yy < tile_h; yy += blockDim.y) {
                int h_in = h_out_base + yy;
                for (int xx = tw; xx < tile_w; xx += blockDim.x) {
                    int w_in = w_out_base + xx;
                    float val = 0.0f;
                    if (b < B && h_in < H && w_in < W) {
                        // x layout: [B, C, H, W]
                        val = x[b * (C * H * W) +
                                c * (H * W) +
                                h_in * W +
                                w_in];
                    }
                    sX[c][yy][xx] = val;
                }
            }
        }

        // Wait for all threads to finish loading weights and inputs before computation
        __syncthreads();

        // 3. Compute Convolution (Compute M outputs for this b, h_out, w_out)
        // Each thread computes the convolution for all M filters at its (h_out, w_out) location
        if (b < B && h_out < H_out && w_out < W_out) {
            // "Temp" registers to hold the accumulated results for all M filters
            float acc[M] = {0.0f};

            // Convolution sum over channels (C) and spatial filter dims (KxK)
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < K; ++p) {
                    for (int q = 0; q < K; ++q) {
                        
                        // +p and +q is the "slide" over the input tile
                        float x_val = sX[c][th + p][tw + q];
                        
                        // For all M filters, accumulate the product
                        for (int m = 0; m < M; ++m) {
                            acc[m] += sW[m][c][p][q] * x_val;
                        }
                    }
                }
            }

            // 4. Write the Result to Global Memory (Output Y)
            // For all M filters, write the accumulated result
            for (int m = 0; m < M; ++m) {
                // Store the accumulated result into the 4D output tensor Y
                y[
                    (b) * (M * H_out * W_out) + 
                    (m) * (H_out * W_out) + 
                    (h_out) * (W_out) + 
                    w_out
                ] = acc[m];
            }
        }
    }

    // TILE_M = Height of the W tile, blockDim.y
    // TILE_N = Width of the X tile, blockDim.x
    // TILE_K = Inner dimension tile size
    template <int B, int M, int C, int H, int W, int K, int H_out, int W_out, int TILE_M, int TILE_K, int TILE_N>
    __global__ void implicitUnrollTiledGemmConv(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y) {

        // Define the shared memory arrays for W (Filter) and X (Activation Patch)
        // NOTE: Because TILE_M, TILE_K, TILE_N could take on different values, we need to consider multiple cases when loading tiles.
        // NOTE: We launch the kernel with blockDim.x = TILE_N and blockDim.y = TILE_M
        // CASE 1: TILE_M > TILE_K:
        //      When loading tileX, we need '''if (threadIdx.y < TILE_K)''' to avoid out-of-bounds access
        // CASE 2: TILE_M < TILE_K:
        //      When loading tileX, we need '''for (int i=threadIdx.y; i<TILE_K; i+=TILE_M)''' to cover all rows in tileX
        // CASE 3: TILE_M == TILE_K:
        //      Normal loading without any special handling but loop/if conditions still work correctly (hope for compiler optimization)
        // CASE 4: TILE_N > TILE_K:
        //      When loading tileW, we need '''if (threadIdx.x < TILE_K)''' to avoid out-of-bounds access
        // CASE 5: TILE_N < TILE_K:
        //      When loading tileW, we need '''for (int i=threadIdx.x; i<TILE_K; i+=TILE_N)''' to cover all columns in tileW
        // CASE 6: TILE_N == TILE_K:
        //      Normal loading without any special handling but loop/if conditions still work correctly (hope for compiler optimization)

        // IMPORTANT: In general, because of the shape of the problem, the most common case is TILE_M < TILE_K < TILE_N (cases 2 and 4)
        // Therefore, conditions applied/implemented below are optimized for this common case.
        // TODO: Maybe consider removing checks all cases and launching with specific TILE sizes that avoid those cases.

        // TODO: Look into padding shared memory to avoid bank conflicts
        __shared__ float tileW[TILE_M][TILE_K];
        __shared__ float tileX[TILE_K][TILE_N];

        // 2. Hoist the Math (Pre-calculate indices)
        // The compiler knows B, H_out, W_out are constants.
        // It turns these divisions into fast bit-shifts or multiplications!
        int col = blockDim.x * blockIdx.x + threadIdx.x;
        
        // NOTE: This replaces calculations inside the tile loading loops! See commented code inside the loop for those constants to make more sense.
        // Constant folding for B, H_out, W_out, and other constants
        int row = blockDim.y * blockIdx.y + threadIdx.y; // No constant that depends on row
        const int KK = K * K;
        const int HW_out = H_out * W_out;

        // Constants that depend on row
        int m_cached = row;

        // Constants that depend on col
        int b_cached       = col / HW_out;
        int hw_out_idx     = col % HW_out;
        int h_out_cached   = hw_out_idx / W_out;
        int w_out_cached   = hw_out_idx % W_out;

        bool valid_row = row < M;
        bool valid_col = col < (B * HW_out);
        
        // The loop iterates over the inner dimension (K dimension = C*K*K) in tiles.
        const int num_w_cols = C * KK;
        float temp = 0.0f; // Accumulator for the output value
        for (int tile_idx = 0; tile_idx < (num_w_cols + TILE_K - 1) / TILE_K; tile_idx++) {
            
            // 1. Load W_unrolled Tile into Shared Memory (tileW)
            // Unroll W on-the-fly from (M, C, K, K) to (M, C*K*K). This is matrix 'A' in GEMM.
            // W is an M x K matrix. Load the tile corresponding to output rows 'row' and inner dim 'tile_idx'

            // w_inner_idx specifies the column within the tile being loaded
            int w_inner_idx = tile_idx * TILE_K + threadIdx.x;
            
            // The last check if for CASE 4 above
            if (valid_row && w_inner_idx < num_w_cols && threadIdx.x < TILE_K) {
                // W is stored in (M, C, K, K) format, so CKK is contiguous in memory.
                // Load W[row, w_inner_idx] from global memory, no need to decode (c, p, q) here.

                tileW[threadIdx.y][threadIdx.x] = w[m_cached * num_w_cols + w_inner_idx];
            }
            else if (threadIdx.x < TILE_K) {
                tileW[threadIdx.y][threadIdx.x] = 0.0f;
            }
        
            // 2. Load X_unrolled Tile into Shared Memory (tileX) ---
            // Unroll X on-the-fly from (B, C, H, W) to (C*K*K, B*H_out*W_out). This is matrix 'B' in GEMM.
            // X_unrolled is a K x N matrix. Load the tile corresponding to inner dim 'tile_idx' and output columns 'col'.
            
            // Loop is for CASE 2 above
            for (int idy = threadIdx.y; idy < TILE_K; idy += TILE_M) {
                // x_inner_idx specifies the row within the tile being loaded
                int x_inner_idx = tile_idx * TILE_K + idy;

                if (x_inner_idx < num_w_cols && valid_col) {

                    // Decode the x_inner_idx to get (c, p, q), which are the channel and filter indices
                    // The dimensions are nested as: c (slowest changing) --> p (medium) --> q (fastest changing).
                    // Below c is K*K dims, below p is K dims
                    int c = x_inner_idx / KK; // Get the channel index by dividing x_inner_idx by the dims under it (K*K)
                    int pq_idx = x_inner_idx % KK; // Get the index within the K*K block
                    int p = pq_idx / K; // Get the filter row index by dividing the index within K*K by the dims under it (K)
                    int q = pq_idx % K; // Get the filter column index by taking modulus K


                    // NOTE: Using pre-calculated spatial indices from constant folding abov, but logic is below for reference
                    // // Decode the column to get (b, h_out, w_out), which are the batch index and output spatial indices
                    // // The dimensions are nested as: b (slowest changing) --> h_out (medium) --> w_out (fastest changing).
                    // // Below b is H_out*W_out dims, below h_out is W_out dims                
                    // int b = col / (H_out * W_out); // Get the batch index by dividing col by the dims under it (H_out*W_out)
                    // int hw_out_idx = col % (H_out * W_out); // Get the index within the H_out*W_out block
                    // int h_out = hw_out_idx / W_out; // Get the output height index by dividing the index within H_out*W_out by the dims under it (W_out)
                    // int w_out = hw_out_idx % W_out; // Get the output width index by taking modulus W_out

                    // Calculate the corresponding input spatial indices to pull from x
                    // At this point, the indices b and c are already correct. 
                    // The output position (h_out, w_out) indicates where the top-left corner of the filter is positioned on the input image (see Lecture 16 slide 59). 
                    // The filter offset (p,q) tells us which element within that patch we are looking for.
                    int h_in = h_out_cached + p;
                    int w_in = w_out_cached + q;
                    
                    if (b_cached < B && c < C && h_in < H && w_in < W) {
                        tileX[idy][threadIdx.x] = x[(b_cached) * (C * H * W) + (c) * (H * W) + (h_in) * (W) + w_in];
                    } else {
                        tileX[idy][threadIdx.x] = 0.0f;
                    }
                
                } else {
                    tileX[idy][threadIdx.x] = 0.0f;
                }
            }
            
            __syncthreads();

            // 3. Compute the Partial Product (accumulate into temp)
            for (int n = 0; n < TILE_K; n++) {
                // W[row, n] * X[n, col]
                temp += tileW[threadIdx.y][n] * tileX[n][threadIdx.x];                 
            }

            // Wait for all threads to finish computing before loading new data that will overwrite shared memory
            __syncthreads();
        }
        
        // 4. Write the Result to Global Memory (Output Y)
        // Write the result to global memory if within bounds
        if (valid_row && valid_col) {

            // Same logic as above to decode row and col back to (b, m, h_out, w_out)
            // Same row decoding as W and same col decoding as X_unrolled
            
            // int m = row;
            
            // int b = col / (H_out * W_out);
            // int hw_out_idx = col % (H_out * W_out);
            // int h_out = hw_out_idx / W_out;
            // int w_out = hw_out_idx % W_out;
            
            // Write the accumulated result into the 4D output tensor Y.
            y[(b_cached) * (M * H_out * W_out) + (m_cached) * (H_out * W_out) + (h_out_cached) * (W_out) + w_out_cached] = temp;
        }
    }      
    
} // namespace eecs471
