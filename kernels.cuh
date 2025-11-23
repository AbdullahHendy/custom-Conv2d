#pragma once

namespace eecs471 {

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
