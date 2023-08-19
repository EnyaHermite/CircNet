import warnings
warnings.filterwarnings("ignore")
import argparse, time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from datetime import datetime
from dataset import MyDataset
from loss import MyLoss
from anchor import MyAnchor
from augmentor import Augment
from CircNet import CircNet
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--dataset', default='../data/voxelized_ABC_0.01', help='which dataset to use [default: ../data/voxelized_ABC_0.01]')
parser.add_argument('--log_dir', default='../log_abc', help='Log dir [default: ../log_abc]')
parser.add_argument('--dimension', type=int, default=3, help='points dimension [default: 3D]')
parser.add_argument('--knn', type=int, default=50, help='number of neighbors [default: 50]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--max_epoch', type=int, default=301, help='Epoch to run [default: 301]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--patch_size', type=int, default=200, help='Number of knn patches sampled from one mesh [default: 200]')
parser.add_argument('--confidence', type=float, default=0.7, help='train center classification threshold')
parser.add_argument('--L_embed', type=int, default=16, help='positional encoding dimension')
parser.add_argument('--s', type=int, default=2, help='number of predicted centers per cell')
parser.add_argument('--alpha1', type=float, default=1.0, help='regression loss balance term')
parser.add_argument('--alpha2', type=float, default=1.0, help='classification loss balance term')
opt = parser.parse_args()

GPU_INDEX = opt.gpu
BATCH_SIZE = opt.batch_size
MAX_EPOCH = opt.max_epoch

# set gpu
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[GPU_INDEX], 'GPU')
tf.config.experimental.set_memory_growth(gpus[GPU_INDEX], True)

LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system('cp %s %s'%(os.path.basename(__file__), LOG_DIR))   # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

L_embed = opt.L_embed
def posenc(x):
    rets = [x]
    for i in range(L_embed):
        for fn in [tf.sin, tf.cos]:
            rets.append(fn(2.**i * x))
    return tf.concat(rets, -1)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def parse_fn(item, dim, knn, is_training=None):
    mesh = o3d.io.read_triangle_mesh(item, enable_post_processing=False)
    points = tf.constant(mesh.vertices, tf.float32)
    triangles = tf.constant(mesh.triangles, tf.int32)

    points = tf.reshape(points, [-1, dim])           # points in space
    triangles = tf.reshape(triangles, [-1, 3])       # triangle list
    triangles = triangles - tf.reduce_min(triangles) # shift index to start from 0

    # normalize points into the unit sphere
    pt_center = (tf.reduce_max(points, axis=0, keepdims=True) +
                 tf.reduce_min(points, axis=0, keepdims=True))/2
    scale = tf.sqrt(tf.reduce_max(tf.reduce_sum((points - pt_center)**2, axis=-1)))
    points = points/scale

    points = Augment.jitter_point_cloud(points, sigma=0.001, prob=1.0)
    if is_training:
        points = Augment.random_scale_point_cloud(points, prob=0.7)

    # compute distance matrix and search for knn
    delta = tf.expand_dims(points,axis=1) - tf.expand_dims(points,axis=0)
    dist = tf.sqrt(tf.reduce_sum(tf.square(delta),axis=-1))
    knn_dists, knn_indices = tf.math.top_k(-dist, k=knn, sorted=True)  # ascending distances
    knn_dists = -knn_dists

    # build the local geometric features
    features = tf.gather(points, knn_indices) - tf.expand_dims(points, axis=1)

    # normalize using local knn scale
    knn_scale = knn_dists[:,1:2]/0.01 #0.01 is the hyper parameter
    normalized_features = tf.math.divide(features, knn_scale[:, :, None])
    points = tf.math.divide(points, knn_scale)

    # sample 200 knn patches from the whole point cloud
    if is_training:
        sample_indices = tf.random.shuffle(tf.range(0, tf.shape(points)[0], delta=1))[:opt.patch_size]
        normalized_features = tf.gather(normalized_features, sample_indices)
        knn_indices = tf.gather(knn_indices, sample_indices)

        # select gt triangles that contain the sampled center points
        point_mask = tf.scatter_nd(tf.expand_dims(sample_indices, axis=-1), updates=tf.ones_like(sample_indices), shape=tf.shape(points[:,0]))
        triangle_mask_1d = tf.gather(point_mask, tf.reshape(triangles, [-1])) # mask triangle vertices
        triangle_mask = tf.reshape(triangle_mask_1d, [-1, 3])
        triangle_mask = tf.reduce_any(tf.cast(triangle_mask, tf.bool), axis=1)
        triangles = tf.boolean_mask(triangles, mask=triangle_mask)

    normalized_features = posenc(normalized_features)
    return points, normalized_features, knn_indices, knn_scale, triangles


class MyModel(tf.Module):
    def __init__(self, net, loss, evaluate, optimizer, train_loss, train_metrics, train_writer, test_loss, test_metrics, test_writer):
        super(MyModel, self).__init__()
        self.model = net
        self.loss = loss
        self.evaluate = evaluate
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_metrics = train_metrics
        self.train_writer = train_writer
        self.test_loss = test_loss
        self.test_metrics = test_metrics
        self.test_writer = test_writer

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, points, normalized_features, knn_indices, knn_scale, triangles, nv, mf):
        with tf.GradientTape() as tape:
            offsets, logits = self.model(normalized_features, training=tf.constant(True))
            loss, loss_loc, loss_cls, \
            loss_cls_p, loss_cls_n, \
            tri_acc, tri_miou, correct_pred_triangles = self.loss(offsets, logits, points, knn_indices, knn_scale, triangles)
        gradients = tape.gradient(loss, self.trainable_variables)

        # every time optimizer.apply_gradients is applied, optimizer.iterations will be increase by 1
        # and the learning rate schedule is called to update the decayed learning rate
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_metrics[0](loss_loc)
        self.train_metrics[1](loss_cls)
        self.train_metrics[2](loss_cls_p)
        self.train_metrics[3](loss_cls_n)
        self.train_metrics[4](tri_acc)
        self.train_metrics[5](tri_miou)
        self.train_metrics[6](correct_pred_triangles)
        return

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, points, normalized_features, knn_indices, knn_scale, triangles, nv, mf):
        offsets, logits = self.model(normalized_features, training=tf.constant(False))
        t_loss, loss_loc, loss_cls, \
        loss_cls_p, loss_cls_n, \
        tri_acc, tri_miou, correct_pred_triangles = self.loss(offsets, logits, points, knn_indices, knn_scale, triangles)

        self.test_loss(t_loss)
        self.test_metrics[0](loss_loc)
        self.test_metrics[1](loss_cls)
        self.test_metrics[2](loss_cls_p)
        self.test_metrics[3](loss_cls_n)
        self.test_metrics[4](tri_acc)
        self.test_metrics[5](tri_miou)
        self.test_metrics[6](correct_pred_triangles)
        return

    def fit(self, train, test, epochs, manager):
        if opt.ckpt_epoch is not None:
            ckpt_epoch = opt.ckpt_epoch
            checkpoint = os.path.dirname(manager.latest_checkpoint) + '/ckpt-%d'%ckpt_epoch
            ckpt.restore(checkpoint)
            ckpt.step = tf.Variable(ckpt_epoch+1,trainable=False)
            print("Restored from {}".format(checkpoint))
            print('checkpoint epoch is %d'%ckpt_epoch, int(ckpt.optimizer.iterations), int(ckpt.step))
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

        template = 'batches %03d, loss: %3.4f, loc_loss: %3.4f,  cls_loss: %3.4f, ' \
                   'pos_cls_loss: %3.4f, neg_cls_loss: %3.4f, ' \
                   'triangle_acc: %3.4f, triangle_miou: %3.4f, correct_triangles: %3.4f   runtime-per-batch: %3.2f ms'
        train_template = 'training '+template
        test_template  = 'test '+template

        for epoch in range(ckpt_epoch+1, epochs):
            log_string(' ****************************** EPOCH %03d TRAINING ******************************'%(epoch))
            log_string(str(datetime.now()))
            sys.stdout.flush()

            batch_idx, train_time = 0, 0.0
            for points, normalized_features, knn_indices, knn_scale, triangles, nv, mf in train.batch(opt.batch_size, drop_remainder=True):
                now = time.time()
                self.train_step(points, normalized_features, knn_indices, knn_scale, triangles, nv, mf)
                batch_time = (time.time() - now)
                train_time += batch_time
                batch_idx += 1
                with self.train_writer.as_default():
                    tf.summary.scalar('train_loss: ', self.train_loss.result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_loss_loc: ', self.train_metrics[0].result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_loss_cls: ', self.train_metrics[1].result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_loss_cls_pos: ', self.train_metrics[2].result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_loss_cls_neg: ', self.train_metrics[3].result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_tri_acc: ', self.train_metrics[4].result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_tri_miou: ', self.train_metrics[5].result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_correct_tri: ', self.train_metrics[6].result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('learning_rate: ', optimizer._decayed_lr(tf.float32),
                                      step=int(ckpt.optimizer.iterations))
                train_writer.flush()
                if batch_idx%50==0:
                    log_string(train_template%(batch_idx, self.train_loss.result(), self.train_metrics[0].result(),
                                               self.train_metrics[1].result(), self.train_metrics[2].result(),
                                               self.train_metrics[3].result(), self.train_metrics[4].result(), 
                                               self.train_metrics[5].result(), self.train_metrics[6].result(), train_time/batch_idx*1000))

            save_path = manager.save(ckpt.step)
            ckpt.step.assign_add(1)
            log_string("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            tf.saved_model.save(model, LOG_DIR)

            log_string(' ------------------------------ EPOCH %03d EVALUATION ------------------------------'%(epoch))
            log_string(str(datetime.now()))
            test_batch_idx, test_time = 0, 0.0
            for points, normalized_features, knn_indices, knn_scale, triangles, nv, mf in test.batch(opt.batch_size, drop_remainder=False):
                now = time.time()
                self.test_step(points, normalized_features, knn_indices, knn_scale, triangles, nv, mf)
                batch_time = (time.time() - now)
                test_time += batch_time
                test_batch_idx += 1

            with self.test_writer.as_default():
                tf.summary.scalar('test_loss: ', self.test_loss.result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_loss_loc: ', self.test_metrics[0].result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_loss_cls: ', self.test_metrics[1].result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_loss_cls_pos: ', self.test_metrics[2].result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_loss_cls_neg: ', self.test_metrics[3].result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_tri_acc: ', self.test_metrics[4].result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_tri_miou: ', self.test_metrics[5].result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_correct_tri: ', self.test_metrics[6].result(),
                                  step=int(ckpt.optimizer.iterations))
            test_writer.flush()
            log_string(test_template%(epoch, self.test_loss.result(), self.test_metrics[0].result(),
                                      self.test_metrics[1].result(), self.test_metrics[2].result(),
                                      self.test_metrics[3].result(), self.test_metrics[4].result(), 
                                      self.test_metrics[5].result(), self.test_metrics[6].result(), test_time/test_batch_idx*1000))

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_metrics[0].reset_states()
            self.train_metrics[1].reset_states()
            self.train_metrics[2].reset_states()
            self.train_metrics[3].reset_states()
            self.train_metrics[4].reset_states()
            self.train_metrics[5].reset_states()
            self.train_metrics[6].reset_states()
            self.test_loss.reset_states()
            self.test_metrics[0].reset_states()
            self.test_metrics[1].reset_states()
            self.test_metrics[2].reset_states()
            self.test_metrics[3].reset_states()
            self.test_metrics[4].reset_states()
            self.test_metrics[5].reset_states()
            self.test_metrics[6].reset_states()


if __name__=='__main__':
    # load file_lists of train/test split
    Lists = {}
    Lists['train'] = [os.path.join(opt.dataset, line.rstrip()+'.ply') for line
                      in open('../data/abc_train_files.txt')]
    Lists['test'] = [os.path.join(opt.dataset, line.rstrip()+'.ply') for line
                     in open('../data/abc_test_files.txt')]
    Lists['test'] = Lists['test'][:100]
    print(len(Lists['train']), len(Lists['test']))

    # anchor and hyper-parameter initialization
    anchor_delta = {'theta': np.pi/6, 'phi': np.pi/6, 'radius': 0.02}
    anchor_range = {'theta': [-np.pi, np.pi], 'phi': [-np.pi/2, np.pi/2], 'radius': [0, 0.2]}
    Anchor = MyAnchor(anchor_delta, anchor_range)
    anchor_centers, num_anchors = Anchor.init_anchors()
    all_dim_keys = ['x', 'y', 'z', 'theta', 'phi', 'radius']
    anchor_delta_list = [anchor_delta[key] for key in all_dim_keys if key in anchor_delta.keys()] # sorted
    thresholds = {'min_angle': np.pi/12, 'max_angle': np.pi/1.2, 'confidence': opt.confidence}
    DIM, KNN = 3, opt.knn
    num_preds_per_anchor = opt.s
    anchor_type = 'other'  # for non-cartesian

    # load in datasets
    trainSet = MyDataset(Lists['train'], anchor_centers, 3, KNN).shuffle(buffer_size=50000).map(parse_fn, is_training=True)
    testSet  = MyDataset(Lists['test'], anchor_centers, 3, KNN).map(parse_fn, is_training=False)

    # create model & Make a loss object
    meshModel = CircNet(DIM, T=num_anchors, S=num_preds_per_anchor, knn=KNN)
    netLoss  = MyLoss(anchor_delta_list, anchor_centers, num_preds_per_anchor, anchor_type, alpha1=opt.alpha1, alpha2=opt.alpha2,
                        thresholds=thresholds).build_loss
    netEvaluate = None

    # Select the adam optimizer
    starter_learning_rate = 0.001
    decay_steps = 80000
    decay_rate = 0.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  starter_learning_rate, decay_steps, decay_rate)
    optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    # Specify metrics for training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric_1 = tf.keras.metrics.Mean(name='train_loss_loc')
    train_metric_2 = tf.keras.metrics.Mean(name='train_loss_cls')
    train_metric_3 = tf.keras.metrics.Mean(name='train_loss_cls_pos')
    train_metric_4 = tf.keras.metrics.Mean(name='train_loss_cls_neg')
    train_metric_5 = tf.keras.metrics.Mean(name='train_tri_acc')
    train_metric_6 = tf.keras.metrics.Mean(name='train_tri_miou')
    train_metric_7 = tf.keras.metrics.Mean(name='train_correct_tri')
    train_metrics  = [train_metric_1,train_metric_2,train_metric_3,train_metric_4,
                      train_metric_5,train_metric_6,train_metric_7]

    # Specify metrics for testing
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_metric_1 = tf.keras.metrics.Mean(name='test_loss_loc')
    test_metric_2 = tf.keras.metrics.Mean(name='test_loss_cls')
    test_metric_3 = tf.keras.metrics.Mean(name='test_loss_cls_pos')
    test_metric_4 = tf.keras.metrics.Mean(name='test_loss_cls_neg')
    test_metric_5 = tf.keras.metrics.Mean(name='test_tri_acc')
    test_metric_6 = tf.keras.metrics.Mean(name='test_tri_miou')
    test_metric_7 = tf.keras.metrics.Mean(name='test_correct_tri')
    test_metrics  = [test_metric_1,test_metric_2,test_metric_3,test_metric_4,
                     test_metric_5,test_metric_6,test_metric_7]

    # create log directory for train and loss summary
    logdir_train = os.path.join(LOG_DIR, "train/"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    logdir_test = os.path.join(LOG_DIR, "test/"+datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Creates a file writer for the log directory.
    train_writer = tf.summary.create_file_writer(logdir_train)
    test_writer = tf.summary.create_file_writer(logdir_test)

    # Create an instance of the model
    model = MyModel(net=meshModel, loss=netLoss, evaluate=netEvaluate, optimizer=optimizer,
                    train_loss=train_loss, train_metrics=train_metrics, train_writer=train_writer,
                    test_loss=test_loss, test_metrics=test_metrics, test_writer=test_writer)

    # create a checkpoint that will manage all the objects with trackable state
    ckpt = tf.train.Checkpoint(step=tf.Variable(1,trainable=False), optimizer=optimizer, net=meshModel,
                               train_loss=train_loss, train_metrics=train_metrics,
                               test_loss=test_loss, test_metrics=test_metrics)
    manager = tf.train.CheckpointManager(ckpt, LOG_DIR+'/tf_ckpts', max_to_keep=MAX_EPOCH)
    model.fit(train=trainSet, test=testSet, epochs=MAX_EPOCH, manager=manager)
