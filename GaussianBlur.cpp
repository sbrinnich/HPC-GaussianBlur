#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define _USE_MATH_DEFINES

#include "CL/cl.h"
#include "tga.h"
#include <cmath>
#include <malloc.h>
#include <fstream>
#include <string>
#include <iostream>

std::string cl_errorstring(cl_int err)
{
	switch (err)
	{
	case CL_SUCCESS:									return std::string("Success");
	case CL_DEVICE_NOT_FOUND:							return std::string("Device not found");
	case CL_DEVICE_NOT_AVAILABLE:						return std::string("Device not available");
	case CL_COMPILER_NOT_AVAILABLE:						return std::string("Compiler not available");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return std::string("Memory object allocation failure");
	case CL_OUT_OF_RESOURCES:							return std::string("Out of resources");
	case CL_OUT_OF_HOST_MEMORY:							return std::string("Out of host memory");
	case CL_PROFILING_INFO_NOT_AVAILABLE:				return std::string("Profiling information not available");
	case CL_MEM_COPY_OVERLAP:							return std::string("Memory copy overlap");
	case CL_IMAGE_FORMAT_MISMATCH:						return std::string("Image format mismatch");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return std::string("Image format not supported");
	case CL_BUILD_PROGRAM_FAILURE:						return std::string("Program build failure");
	case CL_MAP_FAILURE:								return std::string("Map failure");
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return std::string("Misaligned sub buffer offset");
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return std::string("Exec status error for events in wait list");
	case CL_INVALID_VALUE:                    			return std::string("Invalid value");
	case CL_INVALID_DEVICE_TYPE:              			return std::string("Invalid device type");
	case CL_INVALID_PLATFORM:                 			return std::string("Invalid platform");
	case CL_INVALID_DEVICE:                   			return std::string("Invalid device");
	case CL_INVALID_CONTEXT:                  			return std::string("Invalid context");
	case CL_INVALID_QUEUE_PROPERTIES:         			return std::string("Invalid queue properties");
	case CL_INVALID_COMMAND_QUEUE:            			return std::string("Invalid command queue");
	case CL_INVALID_HOST_PTR:                 			return std::string("Invalid host pointer");
	case CL_INVALID_MEM_OBJECT:               			return std::string("Invalid memory object");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  			return std::string("Invalid image format descriptor");
	case CL_INVALID_IMAGE_SIZE:               			return std::string("Invalid image size");
	case CL_INVALID_SAMPLER:                  			return std::string("Invalid sampler");
	case CL_INVALID_BINARY:                   			return std::string("Invalid binary");
	case CL_INVALID_BUILD_OPTIONS:            			return std::string("Invalid build options");
	case CL_INVALID_PROGRAM:                  			return std::string("Invalid program");
	case CL_INVALID_PROGRAM_EXECUTABLE:       			return std::string("Invalid program executable");
	case CL_INVALID_KERNEL_NAME:              			return std::string("Invalid kernel name");
	case CL_INVALID_KERNEL_DEFINITION:        			return std::string("Invalid kernel definition");
	case CL_INVALID_KERNEL:                   			return std::string("Invalid kernel");
	case CL_INVALID_ARG_INDEX:                			return std::string("Invalid argument index");
	case CL_INVALID_ARG_VALUE:                			return std::string("Invalid argument value");
	case CL_INVALID_ARG_SIZE:                 			return std::string("Invalid argument size");
	case CL_INVALID_KERNEL_ARGS:             			return std::string("Invalid kernel arguments");
	case CL_INVALID_WORK_DIMENSION:          			return std::string("Invalid work dimension");
	case CL_INVALID_WORK_GROUP_SIZE:          			return std::string("Invalid work group size");
	case CL_INVALID_WORK_ITEM_SIZE:           			return std::string("Invalid work item size");
	case CL_INVALID_GLOBAL_OFFSET:            			return std::string("Invalid global offset");
	case CL_INVALID_EVENT_WAIT_LIST:          			return std::string("Invalid event wait list");
	case CL_INVALID_EVENT:                    			return std::string("Invalid event");
	case CL_INVALID_OPERATION:                			return std::string("Invalid operation");
	case CL_INVALID_GL_OBJECT:                			return std::string("Invalid OpenGL object");
	case CL_INVALID_BUFFER_SIZE:              			return std::string("Invalid buffer size");
	case CL_INVALID_MIP_LEVEL:                			return std::string("Invalid mip-map level");
	case CL_INVALID_GLOBAL_WORK_SIZE:         			return std::string("Invalid gloal work size");
	case CL_INVALID_PROPERTY:                 			return std::string("Invalid property");
	default:                                  			return std::string("Unknown error code");
	}
}

void checkStatus(cl_int err)
{
	if (err != CL_SUCCESS) {
		printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
		exit(EXIT_FAILURE);
	}
}

void printCompilerError(cl_program program, cl_device_id device)
{
	cl_int status;
	size_t logSize;
	char* log;

	// get log size
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	checkStatus(status);

	// allocate space for log
	log = static_cast<char*>(malloc(logSize));
	if (!log)
	{
		exit(EXIT_FAILURE);
	}

	// read the log
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	checkStatus(status);

	// print the log
	printf("Build Error: %s\n", log);
	free(log);
}

float getGaussianKernelSizeBySigma(float sigma) {
	return 2 * ceil(sigma) + 1;
}

void createGaussianKernel(float outKernel[], float sigma)
{
	float s = 2.0 * sigma * sigma;
	int kernelSize = getGaussianKernelSizeBySigma(sigma);

	// Create Gaussian Kernel
	int r = -sigma;
	float sum = 0.0f;
	for (int i = 0; i < kernelSize; i++) {
		float x = r;
		float v = (exp(-(x * x) / s)) / (sqrt(2.0 * M_PI) * sigma);
		outKernel[i] = v;

		sum += v;
		r++;
	}

	// Normalize distribution
	float div = sum;
	for (int i = 0; i < kernelSize; i++) {
		outKernel[i] /= div;
	}
}

int main() {
	// Read inputs from console
	std::string filenameIn;
	std::string filenameOut;
	float blurSigma;

	printf("Please enter the path to the input file: ");
	std::getline(std::cin, filenameIn);
	printf("Please enter the path to the desired output file (including the desired filename): ");
	std::getline(std::cin, filenameOut);
	printf("Please enter the desired blur strength (floating point value, e.g. 5.0): ");
	std::cin >> blurSigma;

	if (blurSigma < 1) {
		printf("Error: blur strength < 1.0 not possible!\n");
		exit(EXIT_FAILURE);
	}

	// Load test image
	tga::TGAImage* image = new tga::TGAImage();
	bool loaded = tga::LoadTGA(image, filenameIn.c_str());

	if (!loaded) {
		exit(EXIT_FAILURE);
	}

	// input and output arrays
	const unsigned int elementSize = image->width * image->height;
	size_t dataSize = elementSize * sizeof(cl_float4);
	const unsigned int kernelSize = getGaussianKernelSizeBySigma(blurSigma);
	size_t kernelDataSize = kernelSize * kernelSize * sizeof(float);
	cl_float4* vectorIn = static_cast<cl_float4*>(malloc(dataSize));
	float* vectorKernel = static_cast<float*>(malloc(kernelDataSize));
	cl_float4* vectorOut = static_cast<cl_float4*>(malloc(dataSize));

	// Convert image data to float4 vector (r,g,b,a)
	int imageIndexOffset = 3; // rgb image
	if (image->type == 1) {
		imageIndexOffset = 4; // rgba image
	}
	for (unsigned int i = 0; i < elementSize; ++i) {
		float alpha = 1.0f;
		if (image->type == 1) {
			alpha = (float)image->imageData[i * imageIndexOffset + 3];
		}
		vectorIn[i] = { (float)image->imageData[i * imageIndexOffset], (float)image->imageData[i * imageIndexOffset + 1], (float)image->imageData[i * imageIndexOffset + 2], alpha };
	}

	// Create gaussian filter kernel
	createGaussianKernel(vectorKernel, blurSigma);

	// used for checking error status of api calls
	cl_int status;

	// retrieve the number of platforms
	cl_uint numPlatforms = 0;
	checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

	if (numPlatforms == 0)
	{
		printf("Error: No OpenCL platform available!\n");
		exit(EXIT_FAILURE);
	}

	// select the platform
	cl_platform_id platform;
	checkStatus(clGetPlatformIDs(1, &platform, NULL));

	// retrieve the number of devices
	cl_uint numDevices = 0;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

	if (numDevices == 0)
	{
		printf("Error: No OpenCL device available for platform!\n");
		exit(EXIT_FAILURE);
	}

	// select the device
	cl_device_id device;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

	// create context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkStatus(status);

	// create command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
	checkStatus(status);

	// allocate two input and one output buffer for the three vectors
	cl_mem bufferIn = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
	checkStatus(status);
	cl_mem bufferKernel = clCreateBuffer(context, CL_MEM_READ_ONLY, kernelDataSize, NULL, &status);
	checkStatus(status);
	cl_mem bufferOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
	checkStatus(status);

	// write data from the input vectors to the buffers
	checkStatus(clEnqueueWriteBuffer(commandQueue, bufferIn, CL_TRUE, 0, dataSize, vectorIn, 0, NULL, NULL));
	checkStatus(clEnqueueWriteBuffer(commandQueue, bufferKernel, CL_TRUE, 0, kernelDataSize, vectorKernel, 0, NULL, NULL));

	// read the kernel source
	const char* kernelFileName = "kernel.cl";
	std::ifstream ifs(kernelFileName);
	if (!ifs.good())
	{
		printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
		exit(EXIT_FAILURE);
	}

	std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	const char* programSourceArray = programSource.c_str();
	size_t programSize = programSource.length();

	// create the program
	cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char**>(&programSourceArray), &programSize, &status);
	checkStatus(status);

	// build the program
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		printCompilerError(program, device);
		exit(EXIT_FAILURE);
	}

	// define an index space of work-items for execution
	cl_uint maxWorkItemDimensions;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL));

	size_t* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes, NULL));

	if (image->width > maxWorkItemSizes[0] || image->height > maxWorkItemSizes[1])
	{
		printf("Error: Too many elements to process - maximum elements allowed: %zu pixel width and %zu pixel height\n", maxWorkItemSizes[0], maxWorkItemSizes[1]);
		exit(EXIT_FAILURE);
	}

	free(maxWorkItemSizes);
	size_t globalWorkSize[2] = { static_cast<size_t>(image->width), static_cast<size_t>(image->height) };
	size_t localWorkSize[2] = { 16, 16 };

	// create the gaussian blur kernel
	cl_kernel kernel = clCreateKernel(program, "gaussian_blur", &status);
	checkStatus(status);

	// set the kernel arguments
	cl_short isXAxis = 1;
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferIn));
	checkStatus(clSetKernelArg(kernel, 1, 4 * sizeof(float) * (localWorkSize[0] + (kernelSize - 1)) * (localWorkSize[1] + (kernelSize - 1)), 0));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(unsigned int), &(image->width)));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(unsigned int), &(image->height)));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferKernel));
	checkStatus(clSetKernelArg(kernel, 5, sizeof(const unsigned int), &(kernelSize)));
	checkStatus(clSetKernelArg(kernel, 6, sizeof(unsigned short), &isXAxis));
	checkStatus(clSetKernelArg(kernel, 7, sizeof(cl_mem), &bufferOut));

	// execute the kernel
	checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

	// read the device output buffer to the host output array
	checkStatus(clEnqueueReadBuffer(commandQueue, bufferOut, CL_TRUE, 0, dataSize, vectorOut, 0, NULL, NULL));

	// write data from the out vector to the buffer to be able to run the kernel again in the other direction
	checkStatus(clEnqueueWriteBuffer(commandQueue, bufferIn, CL_TRUE, 0, dataSize, vectorOut, 0, NULL, NULL));

	// set the kernel arguments
	isXAxis = 1;
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferIn));
	checkStatus(clSetKernelArg(kernel, 1, 4 * sizeof(float) * (localWorkSize[0] + (kernelSize - 1)) * (localWorkSize[1] + (kernelSize - 1)), 0));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(unsigned int), &(image->width)));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(unsigned int), &(image->height)));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferKernel));
	checkStatus(clSetKernelArg(kernel, 5, sizeof(const unsigned int), &(kernelSize)));
	checkStatus(clSetKernelArg(kernel, 6, sizeof(unsigned short), &isXAxis));
	checkStatus(clSetKernelArg(kernel, 7, sizeof(cl_mem), &bufferOut));

	// execute the kernel
	checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL));

	// read the device output buffer to the host output array
	checkStatus(clEnqueueReadBuffer(commandQueue, bufferOut, CL_TRUE, 0, dataSize, vectorOut, 0, NULL, NULL));

	// Transform output data from float4 back to unsigned char vector
	std::vector<unsigned char> outImageData;
	for (int i = 0; i < elementSize; i++) {
		outImageData.push_back(static_cast<unsigned char>(vectorOut[i].s[0]));
		outImageData.push_back(static_cast<unsigned char>(vectorOut[i].s[1]));
		outImageData.push_back(static_cast<unsigned char>(vectorOut[i].s[2]));
		if (image->type == 1) { // rgba image also needs alpha value
			outImageData.push_back(static_cast<unsigned char>(vectorOut[i].s[3]));
		}
	}

	// save result image
	tga::TGAImage outImage = tga::TGAImage();
	outImage.imageData = outImageData;
	outImage.bpp = image->bpp;
	outImage.width = image->width;
	outImage.height = image->height;
	outImage.type = image->type;
	tga::saveTGA(outImage, filenameOut.c_str());

	// release allocated resources
	free(vectorOut);
	free(vectorKernel);
	free(vectorIn);

	// release opencl objects
	checkStatus(clReleaseKernel(kernel));
	checkStatus(clReleaseProgram(program));
	checkStatus(clReleaseMemObject(bufferOut));
	checkStatus(clReleaseMemObject(bufferKernel));
	checkStatus(clReleaseMemObject(bufferIn));
	checkStatus(clReleaseCommandQueue(commandQueue));
	checkStatus(clReleaseContext(context));
	checkStatus(clReleaseDevice(device));

	exit(EXIT_SUCCESS);
}
