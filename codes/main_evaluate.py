import warnings
warnings.filterwarnings("ignore")
import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import open3d as o3d
from anchor import MyAnchor
from sklearn.neighbors import NearestNeighbors
from dual_extract_triangles import MyEvaluate
from CircNet import CircNet

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--pcd_dir', default='../data/demo/pcd', help='Input point clouds directory [default: ../data/demo/pcd]')
parser.add_argument('--log_dir', default='../log_abc', help='Training log directory [default: ../log_abc]')
parser.add_argument('--write_dir', default='../results/demo', help='Directory to save primitive mesh [default: ../results/demo]')
parser.add_argument('--knn', type=int, default=50, help='Number of neighbors [default: 50]')
parser.add_argument('--ckpt_epoch', type=int, default=300, help='Epoch model to load [default: 300]')
parser.add_argument('--confidence', type=float, default=0.8, help='Confidence to extract triangles [default: 0.8]')
opt = parser.parse_args()

# set gpu
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu], True)


if not os.path.exists(opt.write_dir):
    os.makedirs(opt.write_dir)
if not os.path.exists(opt.write_dir+'/primitive'):
    os.makedirs(opt.write_dir+'/primitive')  # to store the predicted mesh by CircNet

L_embed = 16
def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in [tf.sin, tf.cos]:
            rets.append(fn(2.**i * x))
    return tf.concat(rets, -1)

def parse_fn(item, knn):
    # load the point cloud or mesh vertices
    pcd = o3d.io.read_point_cloud(item) 
    points = np.asarray(pcd.points).astype('float32')
    points = np.unique(points, axis=0) # remove duplicate points
    points = tf.constant(points)
    points = tf.reshape(points, [-1, 3])

    # normalzie points into the unit sphere
    pt_center = (tf.reduce_max(points, axis=0, keepdims=True)+tf.reduce_min(points, axis=0, keepdims=True))/2
    scale = tf.sqrt(tf.reduce_max(tf.reduce_sum((points - pt_center)**2, axis=-1)))
    points = points/scale

    neigh = NearestNeighbors(n_neighbors=knn)
    neigh.fit(points.numpy())
    knn_dists, knn_indices = neigh.kneighbors(points.numpy(), return_distance=True)
    knn_dists = tf.constant(knn_dists,dtype=np.float32)
    knn_indices = tf.constant(knn_indices,dtype=np.int32)

    # build the local geometric features
    features = tf.gather(points, knn_indices) - tf.expand_dims(points, axis=1)
    knn_scale = knn_dists[:, 1:2]/0.01
    normalized_features = tf.math.divide(features, knn_scale[:, :, None])
    points = tf.math.divide(points, knn_scale)
    normalized_features = posenc(normalized_features)
    return points, normalized_features, knn_indices, knn_scale, scale


class MyModel(tf.Module):
    def __init__(self, net, evaluate):

        super(MyModel, self).__init__()
        self.model = net
        self.evaluate = evaluate

    def evaluation(self, points, normalized_features, knn_indices, knn_scale, mesh_fname, scale):
        offsets, logits = self.model(normalized_features, training=tf.constant(False))
        pred_triangles, pred_confidences = self.evaluate.get_predictions(offsets, logits, points, knn_indices, knn_scale)
        points = tf.math.multiply(points, knn_scale)
        points = points * scale

        pred_mesh = o3d.geometry.TriangleMesh()
        pred_mesh.vertices = o3d.utility.Vector3dVector(points.numpy())
        pred_mesh.triangles = o3d.utility.Vector3iVector(pred_triangles.numpy())

        write_mesh_path = '%s/primitive/'%(opt.write_dir) + (mesh_fname.split('/')[-1])
        o3d.io.write_triangle_mesh(write_mesh_path, pred_mesh, write_ascii=True)
        return

    def fit(self, test_list, manager):
        if opt.ckpt_epoch is not None:
            ckpt_epoch = opt.ckpt_epoch
            checkpoint = os.path.dirname(manager.latest_checkpoint) + '/ckpt-%d'%ckpt_epoch
            ckpt.restore(checkpoint)
            ckpt.step = tf.Variable(ckpt_epoch+1,trainable=False)
            print("Restored from {}".format(checkpoint))
        else:
            ckpt.restore(manager.latest_checkpoint)
            print('ckpt.step=%d'%ckpt.step)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
                ckpt_epoch = int(manager.latest_checkpoint.split('-')[-1])
                ckpt.step = tf.Variable(ckpt_epoch+1,trainable=False)
            else:
                print("Initializing from scratch.")
                ckpt_epoch = 0

        print('Number of files:', len(test_list))
        for mesh_fname in tqdm(test_list):
            points, normalized_features, knn_indices, knn_scale, scale = parse_fn(mesh_fname, knn=opt.knn)
            self.evaluation(points, normalized_features, knn_indices, knn_scale, mesh_fname, scale)


if __name__=='__main__':

    anchor_delta = {'theta': np.pi/6, 'phi': np.pi/6, 'radius': 0.02}
    anchor_range = {'theta': [-np.pi, np.pi], 'phi': [-np.pi/2, np.pi/2], 'radius': [0, 0.2]}
    Anchor = MyAnchor(anchor_delta, anchor_range)
    anchor_centers, num_anchors = Anchor.init_anchors()
    all_dim_keys = ['theta', 'phi', 'radius']
    anchor_delta_list = [anchor_delta[key] for key in all_dim_keys if key in anchor_delta.keys()] # sorted
    thresholds = {'min_angle': np.pi/12, 'max_angle': np.pi/1.2, 'confidence': opt.confidence}
    num_preds_per_anchor = 2
    anchor_type = 'other'

    if 'abc' in opt.pcd_dir.lower(): # abc test set
        mesh_list = [os.path.join(opt.pcd_dir, file.strip()+'.ply') for file in open('../data/abc_test_files.txt')]
    else: # demo point clouds, faust dataset
        pcd_paths = opt.pcd_dir
        mesh_list = [file for file in glob.glob(os.path.join(pcd_paths, '*.ply'))]


    CircNetModel = CircNet(3, T=num_anchors, S=num_preds_per_anchor)
    netEvaluate = MyEvaluate(anchor_delta_list, anchor_centers, num_preds_per_anchor, anchor_type,
                             thresholds=thresholds)
    model = MyModel(net=CircNetModel, evaluate=netEvaluate)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1,trainable=False), net=CircNetModel)
    manager = tf.train.CheckpointManager(ckpt, opt.log_dir+'/tf_ckpts', max_to_keep=300)
    model.fit(test_list=mesh_list, manager=manager)


