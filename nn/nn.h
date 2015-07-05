#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

/// query the cpu cores 
#ifdef _WIN32
#include <windows.h>
#elif MACOS
#include <sys/param.h>
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))


/// strurcture for initializing thetas
typedef struct
{
	float *W;
	int  Lin;
	int  Lout;
	float epsilon;
} ThreadData;


class NN
{
public:
	int in_size;
	int out_size;
	int layer_num;	// number of total layers
	int hidden_num;	// number of hidden layers
	int theta_num;	// number of theta arrays
	int thread_num;	// number of pthreads

	float epsilon;

	int   *layer_size;	// size for each layer
	float *input;
	float *output;
	float **hiddenlayers;
	float **thetas;

	NN();	
	~NN();

	static void *PrintHello(void *threadid)
	{
		long tid;
		tid = (long)threadid;
		printf("Hello World! It's me, thread #%ld!\n", tid);
		pthread_exit(NULL);
	}

	/// query the cpu cores
	int queryCores() 
	{
#ifdef WIN32
		SYSTEM_INFO sysinfo;
		GetSystemInfo(&sysinfo);
		return sysinfo.dwNumberOfProcessors;
#elif MACOS
		int nm[2];
		size_t len = 4;
		uint32_t count;

		nm[0] = CTL_HW; nm[1] = HW_AVAILCPU;
		sysctl(nm, 2, &count, &len, NULL, 0);

		if(count < 1) {
			nm[1] = HW_NCPU;
			sysctl(nm, 2, &count, &len, NULL, 0);
			if(count < 1) { count = 1; }
		}
		return count;
#else
		return sysconf(_SC_NPROCESSORS_ONLN);
#endif
	}

	//-----------------------------------------------------------------------//
	// Prepare 
	//-----------------------------------------------------------------------//
	void Init(void);

	//-----------------------------------------------------------------------//
	// Initialize Thetas 
	//-----------------------------------------------------------------------//
	void InitThetas(void);

	void AllocThetas(float *W, int L_in, int L_out)
	{
		uint N = L_in * L_out;
		W = (float *) malloc (sizeof(float) * N); 
	}

#if MT
	static void *RandInitializeWeights(void *ThreadArg)
	{
		ThreadData *current = (ThreadData *) ThreadArg;

		float *W      = current->W;
		int Lin       = current->Lin;
		int Lout      = current->Lout;
		float epsilon = current->epsilon;

		uint N = Lin * Lout;
		for(uint i = 0; i < N; i++)
			W[i] = (static_cast<float>(rand()) / RAND_MAX) * epsilon * 2.f - epsilon;

		pthread_exit(NULL);
	}

#else
	void RandInitializeWeights(float *W, int Lin, int Lout)
	{

		uint N = Lin * Lout;
		for(uint i = 0; i < N; i++)
			W[i] = (static_cast<float>(rand()) / RAND_MAX) * epsilon * 2.f - epsilon;
	}
#endif

};
#endif
