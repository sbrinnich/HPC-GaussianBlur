/*
* A kernel that performs a gaussian blur effect on a given image
*/
__kernel void gaussian_blur(__global float4* in, int width, int height, __global float* gaussianKernel, int gaussianKernelSize, __global float4* out)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	int radius = ((gaussianKernelSize-1)/2);

	float alphaValue = in[x+y*width].w;

	float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	int kernelIndex = 0;
	for(int pixelY = y-radius; pixelY <= y+radius; pixelY++) {
		for(int pixelX = x-radius; pixelX <= x+radius; pixelX++) {
			// Just use nearest pixel when index would be outside of image scope
			int index = min(max(pixelX, 0), width) + min(max(pixelY, 0), height) * width;

			sum += in[index] * gaussianKernel[kernelIndex];
			kernelIndex++;
		}
	}

	sum.w = alphaValue;

	out[x+y*width] = round(sum);
}