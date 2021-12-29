import napari
import numpy as np
import imageio
import elf.segmentation.watershed as ws
import elf.segmentation.features as feats
import elf.segmentation.multicut as mc
from elf.segmentation.features import *
from elf.segmentation.learning import *
from skimage import measure
import time
import nifty
from elf.segmentation.lifted_multicut import lifted_multicut_fusion_moves

def lifted_edges_from_graph_neighborhood(graph, max_graph_distance):
    if max_graph_distance < 2:
        raise ValueError(f"Graph distance must be greater equal 2, got {max_graph_distance}")
    if isinstance(graph, nifty.graph.UndirectedGraph):
        objective = nifty.graph.opt.lifted_multicut.liftedMulticutObjective(graph)
    else:
        graph_ = nifty.graph.undirectedGraph(graph.numberOfNodes)
        graph_.insertEdges(graph.uvIds())
        objective = nifty.graph.opt.lifted_multicut.liftedMulticutObjective(graph_)
    objective.insertLiftedEdgesBfs(max_graph_distance)
    lifted_uvs = objective.liftedUvIds()
    return lifted_uvs

def gene_gt(gt_train):
    GT = []
    offset = 0
    for i in range(gt_train.shape[0]):
        gt = measure.label(gt_train[i], connectivity=1).astype('uint32')
        offset = offset+gt.max()
        gt = ((gt+offset)*gt_train[i]).astype('uint32')
        GT.append(gt)
    GT = np.stack(GT, axis=0)
    return GT

def lifted_edge_label(rag, gt, uv_ids, n_threads=None):
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    node_labels = nrag.gridRagAccumulateLabels(rag, gt, n_threads)
    edge_labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype('uint8')
    return edge_labels

def labels_to_dense_gt( labels_path, probs):
    import vigra.graphs as vgraph
    labels = imageio.volread(labels_path)
    labels = np.transpose(labels,(1,2,0))
    probs = np.transpose(probs, (1, 2, 0))

    gt     = np.zeros_like(labels, dtype = np.uint32)

    offset = 0
    for z in range(gt.shape[2]):

        hmap = vigra.filters.gaussianSmoothing( probs[:,:,z], 2.)
        seeds = vigra.analysis.labelImageWithBackground( labels[:,:,z] )
        gt[:,:,z], _ = vigra.analysis.watershedsNew(hmap, seeds = seeds)
        gt[:,:,z][ gt[:,:,z] != 0 ] += offset
        offset = gt[:,:,z].max()

    # bring to 0 based indexing
    gt -= 1

    # remove isolated segments
    rag_global = vgraph.regionAdjacencyGraph( vgraph.gridGraph(gt.shape[0:3]), gt)

    node_to_node = np.concatenate(
            [ np.arange(rag_global.nodeNum, dtype = np.uint32)[:,None] for _ in range(2)]
            , axis = 1 )

    for z in range(gt.shape[2]):
        rag_local = vgraph.regionAdjacencyGraph( vgraph.gridGraph(gt.shape[0:2]), gt[:,:,z])
        for node in rag_local.nodeIter():
            neighbour_nodes = []
            for nnode in rag_local.neighbourNodeIter(node):
                neighbour_nodes.append(nnode)
            if len(neighbour_nodes) == 1:
                node_coordinates = np.where(gt == node.id)
                if not 0 in node_coordinates[0] and not 511 in node_coordinates[0] and not 0 in node_coordinates[1] and not 511 in node_coordinates[1]:
                    node_to_node[node.id] = neighbour_nodes[0].id

    gt_cleaned = rag_global.projectLabelsToBaseGraph(node_to_node)[:,:,:,0]

    return gt_cleaned

def compute_region_features_(uv_ids, input_map, segmentation, n_threads=None):
    """ Compute edge features from input map accumulated over segmentation
    and mapped to edges.

    Arguments:
        uv_ids [np.ndarray] - edge uv ids
        input_ [np.ndarray] - input data.
        segmentation [np.ndarray] - segmentation.
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads

    # compute the node features
    stat_feature_names = ["Histogram", "Count", "Kurtosis", "Maximum", "Minimum", "Quantiles",
                          "RegionRadii", "Skewness", "Sum", "Variance"]
    coord_feature_names = ["Weighted<RegionCenter>", "RegionCenter"]
    feature_names = stat_feature_names + coord_feature_names
    node_features = vigra.analysis.extractRegionFeatures(input_map, segmentation,
                                                         features=feature_names)

    # get the image statistics based features, that are combined via [min, max, sum, absdiff]
    stat_features = [node_features[fname] for fname in stat_feature_names]
    stat_features = np.concatenate([feat[:, None] if feat.ndim == 1 else feat
                                    for feat in stat_features], axis=1)

    # get the coordinate based features, that are combined via euclidean distance
    coord_features = [node_features[fname] for fname in coord_feature_names]
    coord_features = np.concatenate([feat[:, None] if feat.ndim == 1 else feat
                                     for feat in coord_features], axis=1)

    u, v = uv_ids[:, 0], uv_ids[:, 1]

    # combine the stat features for all edges
    feats_u, feats_v = stat_features[u], stat_features[v]
    features = [np.minimum(feats_u, feats_v), np.maximum(feats_u, feats_v),
                np.abs(feats_u - feats_v), feats_u + feats_v]

    # combine the coord features for all edges
    feats_u, feats_v = coord_features[u], coord_features[v]
    features.append((feats_u - feats_v) ** 2)

    features = np.nan_to_num(np.concatenate(features, axis=1))
    assert len(features) == len(uv_ids)
    return features

def compute_feature(watershed, raw, boundaries):
    if raw.ndim == 2:
        raw = raw[np.newaxis,...]
        watershed = watershed[np.newaxis,...]
        boundaries = boundaries[np.newaxis,...]
    rag = feats.compute_rag(watershed)
    feature1 = compute_boundary_features_with_filters(rag, raw.astype('float32'), apply_2d=True)
    feature1_2 = compute_boundary_features_with_filters(rag, boundaries.astype('float32'), apply_2d=True)
    feature2 = compute_boundary_features(rag, boundaries)
    feature3 = compute_region_features_(rag.uvIds(), raw.astype('float32'), watershed.astype('uint32'))
    features = np.column_stack([feature1, feature1_2, feature2, feature3])
    return features, rag

def test_3D():
    # # # # # # train
    features_total = []
    edge_total = []
    lifted_features_total = []
    lifted_edge_total = []
    # segmentation_2D = np.zeros_like(watershed_test)
    t0 = time.time()
    for z in range(watershed_train.shape[0]):
        # # learning
        features, rag = compute_feature(watershed_train[z], raw_train[z], boundaries_train[z])
        edge_labels = compute_edge_labels(rag, gt_train[z][np.newaxis,...])
        features_total.append(features)
        edge_total.append(edge_labels)
        ### lifted edge
        lifted_edge = lifted_edges_from_graph_neighborhood(rag, 2)
        feature_lifted = compute_region_features_(lifted_edge, raw_train[z][np.newaxis,...].astype('float32'), watershed_train[z][np.newaxis,...].astype('uint32'))
        lifted_edge_labels = lifted_edge_label(rag, gt_train[z][np.newaxis,...], lifted_edge)
        lifted_features_total.append(feature_lifted)
        lifted_edge_total.append(lifted_edge_labels)

    features_total = np.concatenate(features_total,axis=0)
    edge_temp = edge_total.copy()
    edge_total = np.concatenate(edge_temp,axis=0)
    rf = learn_edge_random_forest(features_total, edge_total, oob_score=True, n_estimators=500)
    print('acc of xy:', rf.oob_score_)
    ### lifted edge learning
    lifted_features_total = np.concatenate(lifted_features_total,axis=0)
    lifted_edge_total = np.concatenate(lifted_edge_total,axis=0)
    rf_lifted = learn_edge_random_forest(lifted_features_total, lifted_edge_total, oob_score=True, n_estimators=500)
    print('acc of lifted:', rf_lifted.oob_score_)
    # # # # # # # test
    t1 = time.time()
    features_temp, rag = compute_feature(watershed_test, raw_test, boundaries_test)
    edge_pre = predict_edge_random_forest(rf, features_temp)
    costs = mc.transform_probabilities_to_costs(edge_pre, beta=0.5)
    ### lifted
    lifted_edge = lifted_edges_from_graph_neighborhood(rag, 2)
    feature_lifted = compute_region_features_(lifted_edge, raw_test.astype('float32'),
                                             watershed_test.astype('uint32'))
    edge_pre_lifted = predict_edge_random_forest(rf_lifted, feature_lifted)
    costs_lifted = mc.transform_probabilities_to_costs(edge_pre_lifted, beta=0.5, weighting_exponent=2.)

    print('start rag')
    t2 = time.time()
    print(t1 - t0)
    print(t2 - t1)
    print('start compute')
    t2 = time.time()
    node_labels_lmc = lifted_multicut_fusion_moves(rag, costs, lifted_edge, costs_lifted)
    t3 = time.time()
    print(t3 - t2)
    segmentation3D = feats.project_node_labels_to_pixels(rag, node_labels_lmc)
    # save segmentation
    # for i in range(segmentation3D.shape[0]):
    #     imageio.imwrite('/home/hongb/bigstore/tohongbei/seg3D/' + str(i + 1).zfill(3) + '.tif',
    #                     segmentation3D[i].astype('uint32'))
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw_test, name='raw')
        viewer.add_image(boundaries_test, name='boundaries')
        viewer.add_labels(watershed_test, name='watershed_test')
        viewer.add_labels(segmentation3D, name='multicut3D')

    print('end!')

### load train data
path = r'/home/hongb/bigstore/tohongbei/cmpb-revise/isbi2012/raw_train.tif'
path_pre = r'/home/hongb/bigstore/tohongbei/cmpb-revise/isbi2012/probabilities_train.tif'
path_gt = r'/home/hongb/bigstore/tohongbei/cmpb-revise/isbi2012/groundtruth.tif'
raw_train = imageio.volread(path)
boundaries_train = (imageio.volread(path_pre)/255).astype('float32')
gt_train = labels_to_dense_gt( path_gt, boundaries_train)
gt_train = np.transpose(gt_train, (2, 0, 1))
watershed_train = ws.stacked_watershed(boundaries_train, threshold=.3, sigma_seeds=2., alpha=.9, min_size=20)[0]

## test data
data_path = r'/home/hongb/bigstore/tohongbei/cmpb-revise/isbi2012/raw_test.tif'
label_path = r'/home/hongb/bigstore/tohongbei/cmpb-revise/isbi2012/probabilities_test.tif'
raw_test = imageio.volread(data_path)
boundaries_test = (imageio.volread(label_path)/255).astype('float32')

watershed_test = ws.stacked_watershed(boundaries_test, threshold=.3, sigma_seeds=2., alpha=.9, min_size=20)[0]

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(raw_train, name='raw')
    viewer.add_image(boundaries_train, name='boundaries')
    viewer.add_labels(watershed_train, name='watershed')
    viewer.add_labels(gt_train, name='gt')

### main
test_3D()
