%module fixmesh
%{
  #define SWIG_FILE_WITH_INIT
  #include "fixmesh.h"
%}

%include "numpy.i"
%init %{
import_array();
%}

%apply (int DIM1, float* IN_ARRAY1) {(int Nv, float* vertices)};
%apply (int DIM1, int* IN_ARRAY1) {(int bnd_Nf, int* bnd_triangles)};
%apply (int DIM1, int* IN_ARRAY1) {(int Ne, int* edges)};
%apply (int DIM1, int* IN_ARRAY1) {(int bnd_Ne, int* bnd_indices)};
%apply (int DIM1, int* ARGOUT_ARRAY1) {(int Nadd, int* add_triangles)};

%apply (int DIM1, float* IN_ARRAY1) {(int Nv, float* vertices)};
%apply (int DIM1, int* IN_ARRAY1) {(int Nf, int* sorted_triangles)};
%apply (int DIM1, int* ARGOUT_ARRAY1) {(int Nflag, int* triangle_flags)};

%include "fixmesh.h"
