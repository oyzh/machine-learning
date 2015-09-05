CFLAGS=-g -I.
LFLAGS=

all: facetrain hidtopgm outtopgm facetrain_init0 headtrain

headtrain: headtrain.o imagenet.o backprop_initr.o pgmimage.o
	$(CC) ${LFLAGS} headtrain.o imagenet.o backprop_initr.o pgmimage.o \
	      -o headtrain -lm
facetrain: facetrain.o imagenet.o backprop_initr.o pgmimage.o
	$(CC) ${LFLAGS} facetrain.o imagenet.o backprop_initr.o pgmimage.o \
	      -o facetrain -lm

hidtopgm: hidtopgm.o pgmimage.o backprop_initr.o
	$(CC) ${LFLAGS} hidtopgm.o pgmimage.o backprop_initr.o \
	      -o hidtopgm -lm

outtopgm: outtopgm.o pgmimage.o backprop_initr.o
	$(CC) ${LFLAGS} outtopgm.o pgmimage.o backprop_initr.o \
	      -o outtopgm -lm

facetrain_init0: facetrain.o imagenet.o backprop_init0.o pgmimage.o
	$(CC) ${LFLAGS} facetrain.o imagenet.o backprop_init0.o pgmimage.o \
	      -o facetrain_init0 -lm

backprop_initr.o: backprop.c backprop.h
	$(CC) ${CFLAGS} -c backprop.c
	mv backprop.o backprop_initr.o

backprop_init0.o: backprop.c backprop.h
	$(CC) ${CFLAGS} -c -DINITZERO backprop.c
	mv backprop.o backprop_init0.o
clean:
	rm -r *.o facetrain *~ hidtopgm outtopgm facetrain_init0 headtrain
