/*
 * Author: Tanczos Ladislav
 */
#ifndef CUDA_ALIGN_STRUCTS_H
#define CUDA_ALIGN_STRUCTS_H

#ifdef CUDA_KERNEL
#define CUDA_ALIGN(x)  __align__(x)
#else
#define CUDA_ALIGN(x)
#endif

//Struct name pattern
//S(struct)<variable count>A(align)<align size>

//64bit aligned structure
template <typename S>
struct CUDA_ALIGN(8) CudaS2A8 {
    S v1;
    S v2;
};

//128bit aligned structure
template <typename S>
struct CUDA_ALIGN(16) CudaS2A16 {
    S v1;
    S v2;
};

//128bit aligned structure
template <typename S>
struct CUDA_ALIGN(16) CudaS4A16 {
    S v1;
    S v2;
    S v3;
    S v4;
};

//256bit aligned structure, this perform 2x128bit align
template <typename S>
struct CUDA_ALIGN(16) CudaS8A16 {
    S v1;
    S v2;
    S v3;
    S v4;
    S v5;
    S v6;
    S v7;
    S v8;
};

template <typename S>
struct CUDA_ALIGN(16) CudaS24A16 {
    S v1;
    S v2;
    S v3;
    S v4;
    S v5;
    S v6;
    S v7;
    S v8;
    S v9;
    S v10;
    S v11;
    S v12;
    S v13;
    S v14;
    S v15;
    S v16;
    S v17;
    S v18;
    S v19;
    S v20;
    S v21;
    S v22;
    S v23;
    S v24;
};

template <typename S>
struct CUDA_ALIGN(16) CudaS26A16 {
    S v1;
    S v2;
    S v3;
    S v4;
    S v5;
    S v6;
    S v7;
    S v8;
    S v9;
    S v10;
    S v11;
    S v12;
    S v13;
    S v14;
    S v15;
    S v16;
    S v17;
    S v18;
    S v19;
    S v20;
    S v21;
    S v22;
    S v23;
    S v24;
    S v25;
    S v26;
};

#endif
