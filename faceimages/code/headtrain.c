/*
*****************************************************
*04-Sep-15 Zhenhuan Ouyang, Zhejiang University
*Mechine Learning - Tom M. Mitchell
*section 4
*****************************************************
*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pgmimage.h>
#include <backprop.h>

extern void exit();

int main(int argc, char** argv)
{
  char netname[256], trainname[256], testname[256], predictname[256];
  IMAGELIST *trainlist, *testlist;
  IMAGE *predimg, *iimg;
  int ind, epochs, seed, sumerr, savedelta, list_errors, max, imgsize,epoch,i;
  double out_err,hid_err;
  BPNN *net;

  seed = 904015;
  epochs = 100;
  list_errors = 0;
  netname[0] = trainname[0] = testname[0] = predictname[0] = '\0';

  if (argc < 2)
    {
      /*      printusage(argv[0]);*/
      exit(-1);
    }

  for (ind = 1; ind < argc; ind++)
    {
      if (argv[ind][0] == '-')
	{
	  switch(argv[ind][1])
	    {
	    case 'n': strcpy(netname, argv[++ind]);
	      break;
	    case 'e': epochs = atoi(argv[++ind]);
	      break;
	    case 't': strcpy(trainname, argv[++ind]);
	      break;
	    case 's': strcpy(testname, argv[++ind]);
	      break;
	    case 'p': strcpy(predictname, argv[++ind]);
	      break;
	    default:
	      printf("Unknown switch '%c'\n",argv[ind][1]);
	      break;
	    }
	}
    }

  trainlist = imgl_alloc();
  testlist = imgl_alloc();
  if (trainname[0] != '\0')
    {
      imgl_load_images_from_textfile(trainlist, trainname);
    }
  
  if (testname[0] != '\0')
    {
        imgl_load_images_from_textfile(testlist, testname);
    }


  if (netname[0] == '\0')
    {
      printf("%s: Must specify an output file, i.e., -n <network file>\n", argv[0]);
      exit(-1);
    }

  bpnn_initialize(seed);
  net = bpnn_read(netname);
  if((net == NULL) && (trainname[0] == '\0'))
    {
      printf("Need a network or a trainset\n");
      exit(-1);
    }
  
  if (trainlist->n > 0)
    {
      printf("Creating new network '%s'\n",netname);
      iimg = trainlist->list[0];
      imgsize = ROWS(iimg) * COLS(iimg);
      net = bpnn_create(imgsize, 3, 4);
      for (epoch = 1; epoch <= epochs; epoch++)
	{
	  printf("%d ",epoch);
	  for (i = 0; i < trainlist->n; i++)
	    {
	      load_input_with_image(trainlist->list[i],net);
	      load_target_head(trainlist->list[i], net);
	      bpnn_train(net, 0.3, 0.3, &out_err, &hid_err);
	      sumerr += (out_err + hid_err);
	      printf("%g ",sumerr);
	      printf("\n");
	    }
	}
      bpnn_save(net, netname);
    }

  
  if( (net != NULL) && (predictname[0] != '\0'))
    {
      predimg = img_open(predictname);
      load_input_with_image(predimg,net);
      bpnn_feedforward(net);

      max = 1;
      for(i = 2;i <= net->output_n; i++)
	{
	  if(net->output_units[i] > net->output_units[max])
	    {
	      max = i;
	    }
	}
      if(max == 1)
	printf("Predict image is right");
      else if(max == 2)
	printf("Predict image is stright");
      else if(max == 3)
	printf("Predict image is up");
      else if(max == 4)
	printf("Predict image is left");
    }
  return 0;
}
