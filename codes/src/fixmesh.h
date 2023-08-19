
void edge_manifold(int Nv, float* vertices, int Nf, int* sorted_triangles,
                   int Nflag, int* triangle_flags);

void fill_holes(int Nv, float* vertices, int bnd_Nf, int* bnd_triangles, int Ne, int* edges,
                int bnd_Ne, int* bnd_indices, int Nadd, int* add_triangles);
