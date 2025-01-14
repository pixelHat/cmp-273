LAPACK_LIB=-L/home/hall/Projects/lapack/build/lib/
LAPACK_INCLUDE=-I/home/hall/Projects/lapack/build/include/
LINEAR_ALGEBRA=-llapack -llapacke -lopenblas -lm
BLAS = $(shell pkg-config --libs --cflags openblas)
OPENMP=-fopenmp -L/home/hall/.local/src/OMPT-Examples/INSTALL/lib/ -Wl,--rpath,/home/hall/.local/src/OMPT-Examples/INSTALL/lib/ -I/home/hall/.local/src/OMPT-Examples/INSTALL/include/
CC=clang
# export LD_LIBRARY_PATH=/home/hall/Projects/lapack/build/lib/:$LD_LIBRARY_PATH

libtracer.so: tracer.c tracer.h
	clang -shared -fPIC tracer.c -o libtracer.so

openmp-traced: qr-openmp.c libtracer.so
	 clang -o qr-openmp-traced qr-openmp.c -DOUR_TRACE_INFO -L. $(LAPACK_LIB) $(LAPACK_INCLUDE) $(LINEAR_ALGEBRA) -ltracer -fopenmp

openmp:
	$(CC) -o qr-openmp qr-openmp.c $(LAPACK_LIB) $(LAPACK_INCLUDE) $(BLAS) $(OPENMP) $(LINEAR_ALGEBRA) -Wno-implicit-function-declaration

openmp-verify:
	$(CC) -o qr-openmp-verify qr-openmp.c -DJULIA_OUTPUT $(LAPACK_LIB) $(LAPACK_INCLUDE) $(LINEAR_ALGEBRA)  -fopenmp

seq:
	$(CC) -o qr-seq-tasks qr-openmp.c $(LAPACK_LIB) $(LAPACK_INCLUDE) $(LINEAR_ALGEBRA) -Wno-unknown-pragmas

starpu: qr-starpu.c
	$(CC) -o qr-starpu qr-starpu.c mutils.c $(LAPACK_LIB) $(LAPACK_INCLUDE) $(LINEAR_ALGEBRA) $(BLAS) $(shell pkg-config --libs --cflags starpu-1.4)
