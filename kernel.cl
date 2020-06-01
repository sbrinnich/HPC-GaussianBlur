/*
* A kernel that performs a gaussian blur effect on a given image in one direction (either x or y depending on the isXAxis parameter).
*/
__kernel void gaussian_blur(__global float4* in, __local float4* cached, int width, int height, __global float* gaussianKernel, int gaussianKernelSize, short isXAxis, __global float4* out)
{
	
	// Calculate gaussian kernel radius
	int radius = ((gaussianKernelSize-1)/2);

	// Fetch global position
	int x = get_global_id(0);
	int y = get_global_id(1);
	int globalId = x + y * width;

	// Fetch local position
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	// 2 * radius + get_local_size because the size of the local memory is 2 bigger than the work group size (radius is added left and right)
	int localId = local_x + radius + (local_y + radius) * (get_local_size(0) + 2 * radius); 

	// Cache current pixel
	cached[localId] = in[globalId];

	// Cache additional pixels if needed
	if(isXAxis == 0) {
		if(local_y == 0) {
			int upPixels = min(radius, y);
			
			for(int i = 1; i <= upPixels; i++) {
				cached[local_x + radius + (local_y - i + radius) * (get_local_size(0) + 2 * radius)] = in[x + (y - i) * width];
			}
			for(int i = upPixels + 1; i <= radius; i++) {
				cached[local_x + radius + (local_y - i + radius) * (get_local_size(0) + 2 * radius)] = in[globalId - upPixels];
			}
		} else if (local_y + 1 == get_local_size(1)) {
			int downPixels = min(radius, (height - y - 1));

			for(int i = 1; i <= downPixels; i++) {
				cached[local_x + radius + (local_y + i + radius) * (get_local_size(0) + 2 * radius)] = in[x + (y + i) * width];
			}
			for(int i = downPixels + 1; i <= radius; i++) {
				cached[local_x + radius + (local_y + i + radius) * (get_local_size(0) + 2 * radius)] = in[globalId + downPixels];
			}
		}
	} 
	else 
	{
		if(local_x == 0){
			int leftPixels = min(radius, x);

			for(int i = 1; i <= leftPixels; i++) {
				cached[localId - i] = in[globalId - i];
			}
			for(int i = leftPixels + 1; i <= radius; i++) {
				cached[localId - i] = in[globalId - leftPixels];
			}
		} else if (local_x + 1 == get_local_size(0)) {
			int rightPixels = min(radius, (width - 1 - x));

			for(int i = 1; i <= rightPixels; i++) {
				cached[localId + i] = in[globalId + i];
			}
			for(int i = rightPixels + 1; i <= radius; i++) {
				cached[localId + i] = in[globalId + rightPixels];
			}
		}
	}

	// Sync
	barrier(CLK_LOCAL_MEM_FENCE);

	// Save initial alpha value
	float alphaValue = cached[localId].w;

	// Calculate pixel blur

	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	int kernelIndex = 0;

	//isXAxis is 0 equals FALSE => It is not x axis
	//Otherwise doesn't matter which value, it's always x axis
	if(isXAxis == 0)
	{
		for(int i = -radius; i <= radius; i++) {
			sum += cached[localId + i * (get_local_size(0) + 2 * radius)] * gaussianKernel[kernelIndex];
			kernelIndex++;
		}
	}
	else
	{
		for(int i = -radius; i <= radius; i++) {
			sum += cached[localId + i] * gaussianKernel[kernelIndex];
			kernelIndex++;
		}
	}

	// Reset alpha value to not be affected by blur
	sum.w = alphaValue;

	// Write pixel
	out[globalId] = round(sum);
}