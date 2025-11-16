#include "new_forward.hh"

namespace eecs471 {

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int numARows,
                                    int numAColumns, int numBRows, int numBColumns,
                                    int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this kernel
    // Constant tile size
    const int TILE_SIZE = 32;
    // Define the shared memory arrays for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Define row and column indices
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float temp = 0;
    // for each tile
    for (int k = 0; k < (numAColumns + TILE_SIZE - 1) / TILE_SIZE; k++) {
        // Load data into shared memory keeping in mind boundary conditions for tiles and matrices dimensions
        // Checking for last tile in case A or B "width/cols" is not multiple of TILE_SIZE
        if (k * TILE_SIZE + threadIdx.x < numAColumns && row < numARows)
            tileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + (k * TILE_SIZE + threadIdx.x)];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        if (k * TILE_SIZE + threadIdx.y < numBRows && col < numBColumns)
            tileB[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * numBColumns + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        // Wait for all threads to finish loading data into shared memory
        __syncthreads();

        // Compute the partial product (dot product of the row of A and column of B)
        for (int n = 0; n < TILE_SIZE; n++) {
            temp += tileA[threadIdx.y][n] * tileB[n][threadIdx.x];
        }
        // Wait for all threads to finish computing before loading new data into shared memory
        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = temp;
    }
}

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

// Im2col-style kernel to convert input batch images (B, C, H, W) into 2D column matrix (C*K*K, B*H_out*W_out)
__global__ void x_unroll_kernel(const float *x, float *x_unrolled, const int B, const int C, const int H, const int W, const int K, const int H_out, const int W_out) {
    // Each thread processes one element in the unrolled matrix
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Total number of elements being processed
    int total_elements = C * K * K * B * H_out * W_out;

    // TODO: Check if we coudl get rid of it by launching exact number 
    // No bounds check since we are gonna launch exact number of threads
    if (idx >= total_elements) return;

    // Find row and column that this thread is responsible for. Num of columns = B * H_out * W_out
    // Finding the row and column will give us a "patch" in all images (all channels) in the batch
    // e.g. for idx = 0, we get row = 0, col = 0, which corresponds to the top-left KxK patch of all channels in the first image in the batch
    // See slide 59 in Lecture 16 for visualization
    int row = idx / (B * H_out * W_out);
    int col = idx % (B * H_out * W_out);


    // Decode the row to get (c, p, q), which are the channel and filter indices
    int c = row / (K * K);
    int p = (row / K) % K;
    int q = row % K;

    // Decode the column to get (b, h_out, w_out), which are the batch index and output spatial indices
    int b = col / (H_out * W_out);
    int h_out = (col / W_out) % H_out;
    int w_out = col % W_out;

    // Calculate the corresponding input spatial indices to pull from x
    int h_in = h_out + p;
    int w_in = w_out + q;

    // Copy the value from x to the unrolled matrix
    x_unrolled[idx] = x4d(b, c, h_in, w_in);
}

#undef y4d
#undef x4d
#undef k4d

torch::Tensor forward(const torch::Tensor &x, const torch::Tensor &w, int64_t M) {
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int K = w.size(3);
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Allocate tensor for unrolled x (C*K*K, B*H_out*W_out)
    auto x_unrolled = torch::empty({C * K * K, B * H_out * W_out}, x.options());

    // Launch kernel to unroll x into x_unrolled
    {
        // TODO: Maybe launch exact number of threads instead of doing bounds check in kernel
        // since we know the total number of elements to process in both conv layers
        dim3 gridDim((C * K * K * B * H_out * W_out + 511) / 512);
        dim3 blockDim(512);

        x_unroll_kernel<<<gridDim, blockDim>>>(x.data_ptr<float>(), x_unrolled.data_ptr<float>(), B, C, H, W, K, H_out, W_out);
    }

    // Reshape (unroll) w from (M, C, K, K) to (M, C*K*K)
    auto w_unrolled = w.view({M, C * K * K});

    // Allocate unrolled output y_unrolled (M, B*H_out*W_out) to perform matrix multiplication
    auto y_unrolled = torch::empty({M, B * H_out * W_out}, x.options());

    // Launch tiled matrix multiplication kernel to compute y_unrolled = w_unrolled * x_unrolled
    {
        dim3 gridDim((B * H_out * W_out + 31) / 32, (M + 31) / 32);
        dim3 blockDim(32, 32);

        matrixMultiplyTiled<<<gridDim, blockDim>>>(
            w_unrolled.data_ptr<float>(), x_unrolled.data_ptr<float>(), y_unrolled.data_ptr<float>(),
            M, C * K * K,
            C * K * K, B * H_out * W_out,
            M, B * H_out * W_out);
    }

    // Reshape y_unrolled back to (B, M, H_out, W_out)
    auto y = y_unrolled.view({M, B, H_out, W_out}).permute({1, 0, 2, 3}).contiguous();

    return y;
}
}; // namespace eecs471
