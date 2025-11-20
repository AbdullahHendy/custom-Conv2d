#include "new_forward.hh"

namespace eecs471 {

    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define w4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    #define TILE_SIZE 32

    // Multiplies w_unrolled (M, C*K*K) with x_unrolled (C*K*K, B*H_out*W_out) = (M, B*H_out*W_out)
    // This will produce output y (B, M, H_out, W_out)
    // w_unrolled * x_unrolled = y (x and w are both unrolled on-the-fly inside the kernel)
    __global__ void implicitGemmConv(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                    int B, int C, int H, int W, int K, int H_out, int W_out, int M) {

        // Define the shared memory arrays for W (Filter) and X (Activation Patch)
        __shared__ float tileW[TILE_SIZE][TILE_SIZE];
        __shared__ float tileX[TILE_SIZE][TILE_SIZE];

        int row = blockDim.y * blockIdx.y + threadIdx.y; // Output row index (corresponds to M in W)
        int col = blockDim.x * blockIdx.x + threadIdx.x; // Output col index (corresponds to N in X_unrolled)

        float temp = 0.0f;
        const int num_w_cols = C * K * K; // This is the K dimension in GEMM = num_x_unrolled_rows

        // ---------------------------------------------------------------------------------

        // The loop iterates over the inner dimension (K dimension = C*K*K) in tiles.
        #pragma unroll
        for (int tile_idx = 0; tile_idx < (num_w_cols + TILE_SIZE - 1) / TILE_SIZE; tile_idx++) {
            
            // 1. Load W_unrolled Tile into Shared Memory (tileW)
            // Unroll W on-the-fly from (M, C, K, K) to (M, C*K*K). This is matrix 'A' in GEMM.
            // W is an M x K matrix. Load the tile corresponding to output rows 'row' and inner dim 'tile_idx'

            // w_inner_idx specifies the row within the tile being loaded
            int w_inner_idx = tile_idx * TILE_SIZE + threadIdx.x;
            if (row < M && w_inner_idx < num_w_cols) {
                // Decode the row (m) into (m). Nothing to decode since m is the first dim of W.
                int m = row;

                // Decode the w_inner_idx to get (c, p, q), which are the channel and filter indices
                // The dimensions are nested as: c (slowest changing) --> p (medium) --> q (fastest changing).
                // Below c is K*K dims, below p is K dims
                int c = w_inner_idx / (K * K); // Get the channel index by dividing w_inner_idx by the dims under it (K*K)
                int pq_idx = w_inner_idx % (K * K); // Get the index within the K*K block
                int p = pq_idx / K; // Get the filter row index by dividing the index within K*K by the dims under it (K)
                int q = pq_idx % K; // Get the filter column index by taking modulus K

                tileW[threadIdx.y][threadIdx.x] = w4d(m, c, p, q);
            } else {
                tileW[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // 2. Load X_unrolled Tile into Shared Memory (tileX) ---
            // Unroll X on-the-fly from (B, C, H, W) to (C*K*K, B*H_out*W_out). This is matrix 'B' in GEMM.
            // X_unrolled is a K x N matrix. Load the tile corresponding to inner dim 'tile_idx' and output columns 'col'.
            
            // x_inner_idx specifies the row within the tile being loaded
            int x_inner_idx = tile_idx * TILE_SIZE + threadIdx.y;
            // Check if the current thread is within the bounds of the implicit X_unrolled matrix's K dimension
            if (x_inner_idx < num_w_cols && col < B * H_out * W_out) {
                
                // Decode the x_inner_idx to get (c, p, q), which are the channel and filter indices
                // The dimensions are nested as: c (slowest changing) --> p (medium) --> q (fastest changing).
                // Below c is K*K dims, below p is K dims
                int c = x_inner_idx / (K * K); // Get the channel index by dividing x_inner_idx by the dims under it (K*K)
                int pq_idx = x_inner_idx % (K * K); // Get the index within the K*K block
                int p = pq_idx / K; // Get the filter row index by dividing the index within K*K by the dims under it (K)
                int q = pq_idx % K; // Get the filter column index by taking modulus K


                // Decode the column to get (b, h_out, w_out), which are the batch index and output spatial indices
                // The dimensions are nested as: b (slowest changing) --> h_out (medium) --> w_out (fastest changing).
                // Below b is H_out*W_out dims, below h_out is W_out dims                
                int b = col / (H_out * W_out); // Get the batch index by dividing col by the dims under it (H_out*W_out)
                int hw_out_idx = col % (H_out * W_out); // Get the index within the H_out*W_out block
                int h_out = hw_out_idx / W_out; // Get the output height index by dividing the index within H_out*W_out by the dims under it (W_out)
                int w_out = hw_out_idx % W_out; // Get the output width index by taking modulus W_out


                // Calculate the corresponding input spatial indices to pull from x
                // At this point, the indices b and c are already correct. 
                // The output position (h_out, w_out) indicates where the top-left corner of the filter is positioned on the input image (see Lecture 16 slide 59). 
                // The filter offset (p,q) tells us which element within that patch we are looking for.
                int h_in = h_out + p;
                int w_in = w_out + q;
                
                // Bounds check for the original input X 
                if (b < B && c < C && h_in < H && w_in < W) {
                    // Load the value from X into the Shared Memory tile
                    tileX[threadIdx.y][threadIdx.x] = x4d(b, c, h_in, w_in);
                } else {
                    tileX[threadIdx.y][threadIdx.x] = 0.0f;
                }
            } else {
                // Out of bounds, zero out the tile element
                tileX[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            // Wait for all threads to finish loading data into shared memory
            __syncthreads();

            // 3. Compute the Partial Product (accumulate into temp)
            #pragma unroll
            for (int n = 0; n < TILE_SIZE; n++) {
                // W[row, n] * X[n, col]
                temp += tileW[threadIdx.y][n] * tileX[n][threadIdx.x]; 
            }
            
            // Wait for all threads to finish computing before loading new data
            __syncthreads();
        }

        // 4. Write the Result to Global Memory (Output Y)
        // Write the result to global memory if within bounds
        if (row < M && col < B * H_out * W_out) {

            // Same logic as above to decode row and col back to (b, m, h_out, w_out)
            // Same row decoding as W and same col decoding as X_unrolled
            
            int m = row;
            
            int b = col / (H_out * W_out);
            int hw_out_idx = col % (H_out * W_out);
            int h_out = hw_out_idx / W_out;
            int w_out = hw_out_idx % W_out;
            
            // Write the accumulated result into the 4D output tensor Y.
            y4d(b, m, h_out, w_out) = temp;
        }
    }

    torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
        // // Unchanged Logic
        const int B = x.size(0);
        const int C = x.size(1);
        const int H = x.size(2);
        const int W = x.size(3);
        const int K = w.size(3);
        const int H_out = H - K + 1;
        const int W_out = W - K + 1;

        // Allocate tensor for the final output Y (B, M, H_out, W_out)
        auto y = torch::empty({B, M, H_out, W_out}, x.options());

        // W gets unrolled on-the-fly inside the kernel from (M, C, K, K) to (M, C*K*K)
        // X gets unrolled on-the-fly inside the kernel from (B, C, H, W) to (C*K*K, B*H_out*W_out)

        // Number of rows in the implicit W_unrolled is M (M dimension of A in GEMM)
        const int MM = M;
        // Number of columns in the implicit X_unrolled is B * H_out * W_out (N dimension of B in GEMM)
        const int NN = B * H_out * W_out; 


        // Each block computes a tile of the M x N output matrix.
        dim3 gridDim((NN + TILE_SIZE - 1) / TILE_SIZE, (MM + TILE_SIZE - 1) / TILE_SIZE);
        dim3 blockDim(TILE_SIZE, TILE_SIZE);

        // Launch the implicit GEMM convolution kernel to perform W_unrolled * X_unrolled = Y
        implicitGemmConv<<<gridDim, blockDim>>>(
            w.data_ptr<float>(),
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            B, C, H, W, K, H_out, W_out, M);
        
        // Y is already in the correct shape (B, M, H_out, W_out) because of how it was allocated
        return y;
    }


}; // namespace eecs471
