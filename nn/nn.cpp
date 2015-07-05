#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>	// getopt
#include <pthread.h>

#include "nn.h"

void usage(char *argv0) 
{
	const char *instructions =
		"\nUsage: %s [options]\n\n"
		"    -i size : number of neurons in the input layer\n"
		"    -o size : number of neurons in the output layer\n";
	fprintf(stderr, instructions, argv0);
}

NN::NN() 
{}

NN::~NN()
{
	free(input);
	free(output);
	free(layer_size);

	for(int i = 0; i < hidden_num; i++)
		free(hiddenlayers[i]);
	free(hiddenlayers);

	for(int i = 0; i < theta_num; i++)
		free(thetas[i]);
	free(thetas);
}

//---------------------------------------------------------------------------//
// Initialize NN parameters
//---------------------------------------------------------------------------//
void NN::Init(void)
{
	input  = (float *) malloc (sizeof(float) * in_size);
	output = (float *) malloc (sizeof(float) * out_size);

	/// total layers
	layer_num = hidden_num + 2;

	layer_size = (int *) malloc (sizeof(int) * layer_num);
	layer_size[0]             = in_size;
	layer_size[layer_num - 1] = out_size;
	/// fixme: size of hidden layers
	for(int i = 0; i < hidden_num; i++)
		layer_size[i + 1] = 32;

	/// allocate layers
	hiddenlayers = (float **) malloc (sizeof(float*) * hidden_num);
	for(int i = 0; i < hidden_num; i++)
		hiddenlayers[i] = (float *) malloc (sizeof(float) * layer_size[i + 1]);
		
	/// allocate thetas
	theta_num = hidden_num + 1;
	thetas = (float **) malloc (sizeof(float*) * theta_num);

	int Lin, Lout;
	for(int i = 0; i < theta_num; i++)
	{
		Lin  = layer_size[i];
		Lout = layer_size[i + 1];
		thetas[i] = (float *) malloc (sizeof(float) * Lin * Lout); 
	}

	epsilon = 0.12f;
}



//---------------------------------------------------------------------------//
// Initialize Theta Arrays 
//---------------------------------------------------------------------------//
void NN::InitThetas(void)
{
	/// allocate theta arrray
	for(int i = 0; i < theta_num; i++)	
	{
		AllocThetas(thetas[i],
					layer_size[i],		// Lin
					layer_size[i+1]);	// Lout
	}

	srand(time(NULL));

#if MT
	/// query the cpu cores, where threads <= cores
	int cores = queryCores();

	// pick the smallest one in {cores, thread_num, theta_num}
	int ThreadNum = MIN(MIN(cores, thread_num), theta_num); 

	printf("cpu cores %d, threads %d, theta size %d, launching %d threads\n", 
			cores, 
			thread_num,
			theta_num,
			ThreadNum);
	
	//-------------------------------------------------------------------//
	//	multi threaded
	//-------------------------------------------------------------------//
	pthread_t *threads = (pthread_t *) malloc (sizeof(pthread_t) * ThreadNum);
	pthread_attr_t attr;	
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	/// create thread data structure array
	ThreadData *thread_data = (ThreadData *) malloc (sizeof(ThreadData) * ThreadNum); 

	/// return status
	int rt;

	for(int t = 0; t < ThreadNum; t++)
	{
		/// pass arguments to the thread_data array
		thread_data[t].W         = thetas[t];
		thread_data[t].Lin       = layer_size[t];
		thread_data[t].Lout      = layer_size[t + 1];
		thread_data[t].epsilon   = epsilon; 

		rt = pthread_create(&threads[t], 
		                    NULL, 
							RandInitializeWeights,
							(void *) &thread_data[t]);
							//(void *)(intptr_t) &);
		if (rt){
			printf("ERROR; return code from pthread_create() is %d\n", rt);
			exit(-1);
		}
	}


	for (int t = 0; t < ThreadNum; t++) {
		pthread_join(threads[t], NULL);
	}
	pthread_attr_destroy(&attr);

	//free(threads);
	//free(thread_data);

	/// printf("%f - %f\n", thetas[0][0], thetas[1][0]);

	pthread_exit(NULL);
#else
	//-----------------------------------------------------------------------//
	//	single thread
	//-----------------------------------------------------------------------//
	puts("single thread");
	for(int i = 0; i < theta_num; i++)	
	{
		RandInitializeWeights(thetas[i], 
					layer_size[i],		// Lin
					layer_size[i+1]);	// Lout
	}

	/// printf("%f - %f\n", thetas[0][0], thetas[1][0]);
#endif
}


int main(int argc, char **argv)
{
	//-----------------------------------------------------------------------//
	// Read command line options
	//-----------------------------------------------------------------------//

	/// fixeme: use config file instead
	int opt;
	extern char *optarg;
	
	/// default
	int in_size  = 16;
	int out_size = 16;

	// int hidden_size;

	while ((opt = getopt (argc, argv, "i:o:")) != EOF)
	{
		switch (opt)
		{
			case 'i':
				in_size = atoi(optarg);
				break;

			case 'o':
				out_size = atoi(optarg);
				break;

			case '?':
				usage(argv[0]);
				return 1;

			default:
				usage(argv[0]);
				return 1;
		}
	}
	printf ("input size = %d\n", in_size);
	printf ("output size = %d\n", out_size);
	// printf ("hidden size = %d\n", hidden_size);

	//-----------------------------------------------------------------------//
	// Configure NN 
	//-----------------------------------------------------------------------//
	NN nn;
	nn.in_size    = in_size;
	nn.out_size   = out_size;
	nn.hidden_num = 1;	// number of hidden layer
	nn.thread_num = 3;

	/// initialize parameters
	nn.Init();

	/// initialize thetas
	nn.InitThetas();

	return 0;	
}
