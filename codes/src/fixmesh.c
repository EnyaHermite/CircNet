#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#define PI 3.14159265
#define ANGLE_THRESH 0.5

void get_unit_vector(float vect[])
{
    float norm = vect[0]*vect[0] + vect[1]*vect[1] + vect[2]*vect[2];
    norm = sqrt(norm) + 1.0e-10;

    vect[0] /= norm, vect[1] /= norm, vect[2] /= norm;
    return;
}

float dot_product(const float vect_A[], const float vect_B[])
{
    float dot_P = vect_A[0]*vect_B[0] + vect_A[1]*vect_B[1] + vect_A[2]*vect_B[2];
    return dot_P;
}

void cross_product(const float vect_A[], const float vect_B[], float cross_P[])
{
    cross_P[0] = vect_A[1]*vect_B[2] - vect_A[2]*vect_B[1];
    cross_P[1] = vect_A[2]*vect_B[0] - vect_A[0]*vect_B[2];
    cross_P[2] = vect_A[0]*vect_B[1] - vect_A[1]*vect_B[0];
    return;
}

void get_face_normal(const float* vertices, float face_normal[],
                     const int va, const int vb, const int vc)
{
    float Pa[3] = {vertices[va*3], vertices[va*3+1], vertices[va*3+2]};
    float Pb[3] = {vertices[vb*3], vertices[vb*3+1], vertices[vb*3+2]};
    float Pc[3] = {vertices[vc*3], vertices[vc*3+1], vertices[vc*3+2]};

    float vect_ba[3] = {Pa[0]-Pb[0], Pa[1]-Pb[1], Pa[2]-Pb[2]};
    float vect_bc[3] = {Pc[0]-Pb[0], Pc[1]-Pb[1], Pc[2]-Pb[2]};
    cross_product(vect_ba, vect_bc, face_normal);
    get_unit_vector(face_normal);  // to unit vector

    return;
}

void sort_triangle_vertices(int size, int* array)
{
    assert(size==3);
    int v1 = array[0];
    int v2 = array[1];
    int v3 = array[2];

    int min_v, max_v, mid_v;
    min_v = (v1<v2)?v1:v2;
    min_v = (min_v<v3)?min_v:v3;
    max_v = (v1>v2)?v1:v2;
    max_v = (max_v>v3)?max_v:v3;

    if ((v1!=min_v) & (v1!=max_v))
        mid_v = v1;
    else if ((v2!=min_v) & (v2!=max_v))
        mid_v = v2;
    else if ((v3!=min_v) & (v3!=max_v))
        mid_v = v3;

    array[0] = min_v, array[1] = mid_v, array[2] = max_v;
    return;
}

float cal_edge_length(const float* vertices, const int v1, const int v2)
{
    float x1 = vertices[v1*3];
    float y1 = vertices[v1*3+1];
    float z1 = vertices[v1*3+2];

    float x2 = vertices[v2*3];
    float y2 = vertices[v2*3+1];
    float z2 = vertices[v2*3+2];

    float L12_2 = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2);
    float L12 = sqrt(L12_2);

    return L12;
}

float cal_triangle_angles_var(const float a, const float b, const float c)
{
    float angle_A = acos((b*b+c*c-a*a)/(2*b*c))*180.0/PI;
    float angle_B = acos((c*c+a*a-b*b)/(2*c*a))*180.0/PI;
    float angle_C = acos((a*a+b*b-c*c)/(2*a*b))*180.0/PI;

    float var = (angle_A-60)*(angle_A-60) + (angle_B-60)*(angle_B-60) + (angle_C-60)*(angle_C-60);
    return var;
}

// only for quadrilateral holes
int chooseDiag(const float* vertices, int size, int* array)
{
    assert(size==4);

    int va=array[0], vb=array[1], vc=array[2], vd=array[3];
    int diag_flag; // 1 for (va,vc), 0 for (vb, vd)

    // get edge lengths
    float Lab = cal_edge_length(vertices, va, vb);
    float Lbc = cal_edge_length(vertices, vb, vc);
    float Lcd = cal_edge_length(vertices, vc, vd);
    float Lda = cal_edge_length(vertices, vd, va);
    float Lac = cal_edge_length(vertices, va, vc); // diagonal
    float Lbd = cal_edge_length(vertices, vb, vd); // diagonal

    // average length
    float Tabc = (Lab+Lbc+Lac)/3;
    float Tcda = (Lcd+Lda+Lac)/3;
    float Tabd = (Lab+Lda+Lbd)/3;
    float Tbcd = (Lbc+Lcd+Lbd)/3;

    // compute standard deviation of edge lengths
    float var_abc = (Lab-Tabc)*(Lab-Tabc) + (Lbc-Tabc)*(Lbc-Tabc) + (Lac-Tabc)*(Lac-Tabc);
    float var_cda = (Lcd-Tcda)*(Lcd-Tcda) + (Lda-Tcda)*(Lda-Tcda) + (Lac-Tcda)*(Lac-Tcda);
    float var_abd = (Lab-Tabd)*(Lab-Tabd) + (Lda-Tabd)*(Lda-Tabd) + (Lbd-Tabd)*(Lbd-Tabd);
    float var_bcd = (Lbc-Tbcd)*(Lbc-Tbcd) + (Lcd-Tbcd)*(Lcd-Tbcd) + (Lbd-Tbcd)*(Lbd-Tbcd);
    float _var_ac = var_abc>var_cda?var_abc:var_cda;
    float _var_bd = var_abd>var_bcd?var_abd:var_bcd;

    if (_var_ac<_var_bd)
        diag_flag = 1;
    else
        diag_flag = 0;

    return diag_flag;
}

// check adjacent triangles of edge (v1,v2)
void processAdjacency(const float* vertices, int Nflag, const int* triangles,
                      int* triangle_flags, int num_seeds[],
                      int* seed_triangle_indices, int va, int vb, int vc)
{
    int num_adjs=0, num_adjs_chosen=0;

    float normal_abc[3] = {0.0};
    get_face_normal(vertices, normal_abc, va, vb, vc); // face normal of triangle abc

    // assume each edge is adjacent to 1000 triangles
    int* all_adj_idx_chosen = (int*)calloc(1000, sizeof(int));
    int* all_adj_idx = (int*)calloc(1000, sizeof(int));
    int* all_vd = (int*) calloc(1000, sizeof(int));
    for(int i=0;i<Nflag;i++)
    {
        if (triangle_flags[i]>=0)
        {
            int v1=triangles[3*i], v2=triangles[3*i+1], v3=triangles[3*i+2];

            int vd = -1;
            if ((v1==va) && (v2==vb) && (v3!=vc))
                vd = v3;
            else if ((v2==va) && (v3==vb) && (v1!=vc))
                vd = v1;
            else if ((v1==va) && (v3==vb) && (v2!=vc))
                vd = v2;
            else
                continue;

            all_adj_idx[num_adjs] = i;
            all_vd[num_adjs] = vd;
            num_adjs++;

            if(triangle_flags[i]==0)
                triangle_flags[i] = -1;
            else
            {
                all_adj_idx_chosen[num_adjs_chosen] = i;
                num_adjs_chosen++;
            }
        }
    }

    if (num_adjs_chosen>1) // delete the redundant adjacent triangles
    {
        for(int k=1;k<num_adjs_chosen;k++)
        {
            int i = all_adj_idx_chosen[k];
            triangle_flags[i] = -1;
        }
    }
//    printf("(num_adjs, num_adjs_chosen)=(%d,%d)\n",num_adjs, num_adjs_chosen);

    if (num_adjs_chosen==0)
    {
        float min_alpha = 1.0;
        int best_match_adj = -100;
        for(int k=0;k<num_adjs;k++)
        {
            int vd = all_vd[k];
            float normal_abd[3] = {0.0};
            get_face_normal(vertices, normal_abd, va, vb, vd);
            float alpha = dot_product(normal_abc, normal_abd);

            if ((alpha<ANGLE_THRESH) & (alpha<min_alpha))
            {
                best_match_adj = all_adj_idx[k];
                min_alpha = alpha;
            }
        }
        if(best_match_adj>=0)
        {
            seed_triangle_indices[num_seeds[0]] = best_match_adj;
            triangle_flags[best_match_adj] = 1;
            num_seeds[0]++;
        }
    }
    free(all_adj_idx_chosen);
    free(all_adj_idx);
    free(all_vd);

    return;
}


// complexity no more than O(Nf*Nf)
void edge_manifold(int Nv, float* vertices, int Nf, int* sorted_triangles,
                   int Nflag, int* triangle_flags)
{
    assert((Nf/3)==Nflag);
    for(int i=0;i<Nflag;i++)     // initialize to zero
        triangle_flags[i] = 0;

    int num_seeds[1] = {0};
    int* seed_triangle_indices = (int*)calloc(Nflag, sizeof(int));  // initialize to zero
    triangle_flags[0] = 1;  // set flag=1 for the seed triangle
    seed_triangle_indices[0] = 0;  // seed triangle index

    // grow the manifold mesh from seed triangles
    num_seeds[0] = 1;
    int start_i=0;
    while(start_i>=0)
    {
        for (int i=start_i; i<num_seeds[0]; i++)  // iterate by edge ids
        {
            int seed_id = seed_triangle_indices[i];
            if (triangle_flags[seed_id]==1)
            {
                int va = sorted_triangles[3*seed_id];
                int vb = sorted_triangles[3*seed_id+1];
                int vc = sorted_triangles[3*seed_id+2];

                processAdjacency(vertices, Nflag, sorted_triangles, triangle_flags,
                                 num_seeds, seed_triangle_indices, va, vb, vc);
                processAdjacency(vertices, Nflag, sorted_triangles, triangle_flags,
                                 num_seeds, seed_triangle_indices,vb, vc, va);
                processAdjacency(vertices, Nflag, sorted_triangles, triangle_flags,
                                 num_seeds, seed_triangle_indices, va, vc, vb);
            }
        }

        start_i=-1; // reset
        for(int i=0;i<Nflag;i++)
        {
            if(triangle_flags[i]==0)
            {
                seed_triangle_indices[num_seeds[0]] = i;
                triangle_flags[i] = 1;
                start_i = num_seeds[0];
                num_seeds[0]++;
                break;
            }
        }
    }
    free(seed_triangle_indices);

    return;
}

// complexity of O(Ne*(Ne+Nf))
void fill_holes(int Nv, float* vertices, int bnd_Nf, int* bnd_triangles, int Ne, int* edges,
                int bnd_Ne, int* bnd_indices, int max_added, int* add_triangles)
{
    for(int i=0;i<max_added;i++) // initialize to zero
    {
        add_triangles[i] = 0;
    }
    int Nadd = 0; // number of new triangles added

    int* bnd_flags = (int*)calloc(Ne, sizeof(int));   // initialize to zero
    for (int bnd_i=0; bnd_i<bnd_Ne; bnd_i++)
    {
        int i = bnd_indices[bnd_i];
        bnd_flags[i] = 1;
    }

    // searching for loop holes with O(Ne*(Ne+Nf)) complexity
    for (int bnd_i=0; bnd_i<bnd_Ne; bnd_i++)  // iterate by edge ids
    {
        int i = bnd_indices[bnd_i];
        if (bnd_flags[i]==1)  // yes boundary
        {
            // seed edges
            int v1=edges[i*2], v2=edges[i*2+1];

            int lid = 0;
            int temp_loop[100*6] = {0};        // each as (num_vertices, v1, v2, v3, v4, v5)
            int edge_indices[100*5] = {0};     // store the edge indices to update the bnd_flags
            int* vertex_flags = (int*)calloc(Nv*2, sizeof(int)); // indicate whether vertex is the loop

            vertex_flags[v1*2] = 1;
            vertex_flags[v2*2] = 1;
            vertex_flags[v1*2+1] = -1;
            vertex_flags[v2*2+1] = -1;

            int num_searched = 0;

            // search the loop
            for(int bnd_j=bnd_i+1; bnd_j<bnd_Ne; bnd_j++)
            {
                int j = bnd_indices[bnd_j];
                if (bnd_flags[j]==1)  // yes boundary
                {
                    int va=edges[j*2], vb=edges[j*2+1];

                    if ((vertex_flags[va*2]==0) & (vertex_flags[vb*2]==1))
                    {
                        int temp;
                        temp=va, va=vb, vb=temp;
                    }
                    if ((vertex_flags[va*2]==1) & (vertex_flags[vb*2]==0))
                    {
                        if ((v1==va) || (v2==va)) // grow the seed loop to v1->v2->vb, for quad-loop
                        {
                            int v_begin = (v2==va)?v1:v2;

                            temp_loop[lid*6]=3;
                            temp_loop[lid*6+1]=v_begin;
                            temp_loop[lid*6+2]=va;
                            temp_loop[lid*6+3]=vb;

                            vertex_flags[vb*2] = 1;
                            vertex_flags[vb*2+1] = lid;

                            edge_indices[lid*5] = 2;
                            edge_indices[lid*5+1] = i;
                            edge_indices[lid*5+2] = j;

                            lid++;
                        }
                        else // grow the 3-loop to v1-v2-v3-v4
                        {
                            int curr_id = vertex_flags[va*2+1];
                            assert(curr_id>=0);

                            int num_v = temp_loop[curr_id*6];
                            int v3 = temp_loop[curr_id*6+3];
                            if ((num_v==3) & (v3==va))
                            {
                                temp_loop[curr_id*6]++;
                                temp_loop[curr_id*6+4] = vb;

                                vertex_flags[vb*2] = 1;
                                vertex_flags[vb*2+1] = curr_id;

                                edge_indices[curr_id*5]++;
                                int num_e = edge_indices[curr_id*5];
                                edge_indices[curr_id*5+num_e] = j;
                            }
                        }
                        continue;
                    }

                    if ((vertex_flags[va*2]==1) & (vertex_flags[vb*2]==1))
                    {
                        int Aid = vertex_flags[va*2+1];
                        int Bid = vertex_flags[vb*2+1];
                        assert((Aid>=0) || (Bid>=0));

                        int nv_A = temp_loop[Aid*6];
                        int nv_B = temp_loop[Bid*6];

                        int min_id = Aid<Bid?Aid:Bid;
                        int max_id = Aid>Bid?Aid:Bid;

                        int opt_id = -1;
                        if (min_id==-1)
                        {
                            int num_v = temp_loop[max_id*6];
                            assert((num_v>=3) & (max_id>=0));
                            int v_end = temp_loop[max_id*6+num_v];
                            if (v_end==vb)
                            {
                                int temp;
                                temp=va, va=vb, vb=temp;
                            }
                            temp_loop[max_id*6+num_v+1] = vb; // no update on temp_loop[max_id*6]

                            edge_indices[max_id*5]++;
                            int num_ne = edge_indices[max_id*5];
                            edge_indices[max_id*5+num_ne] = j;

                            opt_id = max_id;
                        }

                        if ((nv_A==3) & (nv_B==3) & (Aid>=0) & (Bid>=0))
                        {
                            int v3_a = temp_loop[Aid*6+3];
                            int v3_b = temp_loop[Bid*6+3];
                            if (v3_b==va)
                            {
                                int temp;
                                temp=va, va=vb, vb=temp;
                            }
                            assert((v3_a==va) & (v3_b==vb));

                            if (temp_loop[min_id*6+1]==temp_loop[max_id*6+2])
                            {
                                temp_loop[min_id*6+4] = temp_loop[max_id*6+3];
                                temp_loop[min_id*6+5] = temp_loop[max_id*6+2];
                                temp_loop[min_id*6] = 4;
                                temp_loop[max_id*6] = 0;

                                int min_ne = edge_indices[min_id*5];
                                int max_ne = edge_indices[max_id*5];

                                assert((min_ne==2) & (max_ne==2));
                                edge_indices[min_id*5+3] = j;
                                edge_indices[min_id*5+4] = edge_indices[max_id*5+2];
                                edge_indices[min_id*5] = 4;

                                opt_id = min_id;
                            }
                        }

                        if (opt_id>=0)  // found closed loop
                        {
                            int loop_size = temp_loop[opt_id*6];
                            int loop[5] = {temp_loop[opt_id*6+1], temp_loop[opt_id*6+2],
                                           temp_loop[opt_id*6+3], temp_loop[opt_id*6+4],
                                           temp_loop[opt_id*6+5]};

                            if ((loop_size==3) & (loop[0]==loop[3]))
                            {
                                int is_exist = 0;
                                for(int k=0; k<bnd_Nf; k++) // check if the triangle already exist
                                {
                                    int tv_1=bnd_triangles[k*3], tv_2=bnd_triangles[k*3+1], tv_3=bnd_triangles[k*3+2];
                                    sort_triangle_vertices(3, loop);
                                    if ((tv_1==loop[0]) & (tv_2==loop[1]) & (tv_3==loop[2]))
                                    {
                                        is_exist = 1;
                                        break;
                                    }
                                }

                                if (is_exist==0)
                                {
                                    add_triangles[3+Nadd*3] = loop[0];
                                    add_triangles[3+Nadd*3+1] = loop[1];
                                    add_triangles[3+Nadd*3+2] = loop[2];
                                    Nadd++;

                                    assert(edge_indices[opt_id*5]==3);
                                    for(int iter=0;iter<edge_indices[opt_id*5];iter++)
                                    {
                                        int e_id = edge_indices[opt_id*5+iter+1];
                                        bnd_flags[e_id] = 0;
                                    }
                                    num_searched++;
                                }
                            }
                            if ((loop_size==4) & (loop[0]==loop[4]))
                            {
                                int is_exist = 0;
                                for(int k=0; k<Ne; k++) // check if the diagonal edge exists as manifold edge
                                {
                                    int e_v1=edges[k*2], e_v2=edges[k*2+1];
                                    if(loop[0]>loop[2])
                                    {
                                        int temp;
                                        temp=loop[0], loop[0]=loop[2], loop[2]=temp;
                                    }
                                    if(loop[1]>loop[3])
                                    {
                                        int temp;
                                        temp=loop[1], loop[1]=loop[3], loop[3]=temp;
                                    }

                                    if (((e_v1==loop[0]) & (e_v2==loop[2])) ||
                                        ((e_v1==loop[1]) & (e_v2==loop[3])))
                                    {
                                        is_exist = 1;
                                        break;
                                    }
                                }

                                if (is_exist==0) {
                                    int diag_flag = chooseDiag(vertices, 4, loop);
                                    if (diag_flag==1) {
                                        add_triangles[3+Nadd*3] = loop[0];
                                        add_triangles[3+Nadd*3+1] = loop[1];
                                        add_triangles[3+Nadd*3+2] = loop[2];
                                        Nadd++;

                                        add_triangles[3+Nadd*3] = loop[0];
                                        add_triangles[3+Nadd*3+1] = loop[2];
                                        add_triangles[3+Nadd*3+2] = loop[3];
                                        Nadd++;
                                    } else {
                                        add_triangles[3+Nadd*3] = loop[0];
                                        add_triangles[3+Nadd*3+1] = loop[1];
                                        add_triangles[3+Nadd*3+2] = loop[3];
                                        Nadd++;

                                        add_triangles[3+Nadd*3] = loop[1];
                                        add_triangles[3+Nadd*3+1] = loop[2];
                                        add_triangles[3+Nadd*3+2] = loop[3];
                                        Nadd++;
                                    }

                                    assert(edge_indices[opt_id*5]==4);
                                    for(int iter=0;iter<edge_indices[opt_id*5];iter++)
                                    {
                                        int e_id = edge_indices[opt_id*5+iter+1];
                                        bnd_flags[e_id] = 0;
                                    }
                                    num_searched++;
                                }
                            }
                            break;
                        }
                    }
                }
            }
            assert((num_searched==0) || (num_searched==1));
            free(vertex_flags);
        }
    }

    free(bnd_flags);
    add_triangles[0] = Nadd;
    return;
}



