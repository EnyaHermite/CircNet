import numpy as np
from . import fixmesh        


class PostProcess:

    def __init__(self, thresh=0.5):
        self.thresh = thresh

    def clean_mesh(self, points, triangles, fill_holes=True):
        points = points.astype("float32")
        triangles = triangles.astype("int32")

        triangles = np.sort(triangles, axis=-1)
        v1 = triangles[:, 0]
        v2 = triangles[:, 1]
        v3 = triangles[:, 2]
        edges = np.reshape(np.stack([v1, v2, v2, v3, v1, v3], axis=1), [-1,2])
        uni_edges, uni_inv, uni_cnts = np.unique(edges, return_inverse=True,
                                                 return_counts=True, axis=0)
        uni_cnts = uni_cnts.astype('int32')
        edge_cnts = uni_cnts[uni_inv]
        triangle_cnts = np.reshape(edge_cnts, [-1,3])
        labels = np.all(triangle_cnts<=2, axis=-1)
        non_manifold_triangles = triangles[~labels,:]
        manifold_triangles = triangles[labels,:]

        triangles = non_manifold_triangles
        if triangles.shape[0]==0:
            return manifold_triangles

        angles = self.compute_angles(points, triangles)*180/np.pi
        angles = np.sort(angles, axis=-1)
        std = np.std(angles-60, axis=-1)
        sort_indices = np.argsort(std)
        triangles = triangles[sort_indices]
        Nf = triangles.shape[0]

        points_1D = np.reshape(points,[-1])
        triangles_1D = np.reshape(triangles,[-1])
        triangle_flags = fixmesh.edge_manifold(points_1D, triangles_1D, Nf)
        keep_triangles = triangles[triangle_flags==1,:]
        keep_triangles = np.concatenate([keep_triangles, manifold_triangles], axis=0)
        if fill_holes:
            add_triangles = self.fill_holes(points, keep_triangles)
            all_triangles = np.concatenate([keep_triangles, add_triangles], axis=0)
        else:
            all_triangles = keep_triangles
        return all_triangles

    def fill_holes(self, points, triangles):
        points = points.astype("float32")
        triangles = triangles.astype("int32")

        triangles = np.sort(triangles, axis=-1)
        v1 = triangles[:, 0]
        v2 = triangles[:, 1]
        v3 = triangles[:, 2]
        edges = np.reshape(np.stack([v1, v2, v2, v3, v1, v3], axis=1), [-1,2])
        uni_edges, uni_inv, uni_cnts = np.unique(edges, return_inverse=True,
                                                 return_counts=True, axis=0)
        edge_cnts = uni_cnts[uni_inv]
        triangle_cnts = np.reshape(edge_cnts, [-1,3])
        bnd_triangles = triangles[np.all(triangle_cnts==1, axis=-1),:]
        bnd_indices = np.where(uni_cnts==1)[0]
        num_bnd_edges = bnd_indices.shape[0]//3

        points = np.reshape(points,[-1])
        bnd_triangles = np.reshape(bnd_triangles,[-1])
        uni_edges = np.reshape(uni_edges,[-1])
        bnd_indices = bnd_indices.astype('int32')
        add_triangles = fixmesh.fill_holes(points, bnd_triangles, uni_edges,
                                           bnd_indices, num_bnd_edges*3+3)
        add_triangles = np.reshape(add_triangles,[-1,3])
        Nadd = add_triangles[0,0]
        add_triangles = add_triangles[1:Nadd+1,:]
        return add_triangles

    @staticmethod
    def check_manifold(triangles):
        v1 = triangles[:,0]
        v2 = triangles[:,1]
        v3 = triangles[:,2]
        edges = np.concatenate([np.stack([v1, v2], axis=1),
                                np.stack([v2, v3], axis=1),
                                np.stack([v1, v3], axis=1)], axis=0)
        edges = np.sort(edges, axis=-1)
        uni_edges, uni_cnts = np.unique(edges, return_counts=True, axis=0)
        uni_degrees, degree_cnts = np.unique(uni_cnts, return_counts=True, axis=0)
        print("edge degree counts: ", uni_degrees, degree_cnts)
        # assert(np.max(uni_degrees)<=2)
        return

    @staticmethod
    def compute_angles(points, triangles):
        A = points[triangles[:,0],:]  # triangle vertex A
        B = points[triangles[:,1],:]  # triangle vertex B
        C = points[triangles[:,2],:]  # triangle vertex C

        # calculate the angles
        l_AB = np.sqrt(np.sum((A - B) ** 2, axis=-1))
        l_AC = np.sqrt(np.sum((A - C) ** 2, axis=-1))
        l_BC = np.sqrt(np.sum((B - C) ** 2, axis=-1))
        dot_A = np.sum((B - A) * (C - A), axis=-1) / (l_AB * l_AC)
        dot_B = np.sum((A - B) * (C - B), axis=-1) / (l_AB * l_BC)
        dot_C = np.sum((A - C) * (B - C), axis=-1) / (l_AC * l_BC)
        dot_A = np.clip(dot_A, -1., 1.)  # to avoid complex values
        dot_B = np.clip(dot_B, -1., 1.)
        dot_C = np.clip(dot_C, -1., 1.)
        angle_A = np.arccos(dot_A)  # [0, PI] in radians
        angle_B = np.arccos(dot_B)
        angle_C = np.arccos(dot_C)
        angles = np.stack([angle_A, angle_B, angle_C], axis=1)
        return angles