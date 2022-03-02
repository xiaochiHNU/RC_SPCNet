import imageio
import numpy as np
import time
from concurrent import futures
import multiprocessing
import nifty
import nifty.graph.rag as nrag
import vigra
import vigra.graphs as graphs
from elf.parallel.relabel import relabel_consecutive
import elf.segmentation.watershed as ws
import elf.segmentation.features as feats
import elf.segmentation.multicut as mc
import elf.segmentation.learning as elearn
from elf.segmentation.lifted_multicut import lifted_multicut_fusion_moves, blockwise_lifted_multicut

def edge_indications_rag(rag):
    n_edges = rag.edgeNum
    edge_indications = np.zeros(n_edges)
    # TODO no loops, no no no loops
    for edge_id in range(n_edges):
        edge_coords = rag.edgeCoordinates(edge_id)
        z_coords = edge_coords[:, 0]
        z = np.unique(z_coords)
        assert z.size == 1, "Edge indications can only be calculated for flat superpixel" + str(z)
        # check whether we have a z or a xy edge
        if z - int(z) == 0.:
            # xy-edge!
            edge_indications[edge_id] = 1
        else:
            # z-edge!
            edge_indications[edge_id] = 0
    return edge_indications

def topology_features(raw, watershed, use_2d_edges):
    assert isinstance(use_2d_edges, bool), type(use_2d_edges)

    if not use_2d_edges:
        n_feats = 1
    else:
        n_feats = 7
    grid = graphs.gridGraph(watershed.shape[0:3])
    rag = graphs.regionAdjacencyGraph(grid, watershed.astype('uint32'))
    n_edges = rag.edgeNum
    topology_features = np.zeros((n_edges, n_feats))

    # length / area of the edge
    edge_lens = rag.edgeLengths()
    assert edge_lens.shape[0] == n_edges
    topology_features[:, 0] = edge_lens
    topology_features_names = ["TopologyFeature_EdgeLength"]

    # deactivated for now, because it segfaults for large ds
    # TODO look into this
    ## number of connected components of the edge
    # n_ccs = self.edge_connected_components(seg_id)
    # assert n_ccs.shape[0] == n_edges
    # topology_features[:,1] = n_ccs
    # topology_features_names = ["TopologyFeature_NumFaces"]

    # extra feats for z-edges in 2,5 d
    if use_2d_edges:

        # edge indications
        edge_indications = edge_indications_rag(rag)
        assert edge_indications.shape[0] == n_edges
        topology_features[:, 1] = edge_indications
        topology_features_names.append("TopologyFeature_xy_vs_z_indication")

        # region sizes to build some features
        statistics = ["Count", "RegionCenter"]

        extractor = vigra.analysis.extractRegionFeatures(
            raw.astype(np.float32),
            watershed.astype(np.uint32),
            features=statistics)

        z_mask = edge_indications == 0

        sizes = extractor["Count"]
        uvIds = rag.uvIds()
        uvIds = np.sort(uvIds, axis = 1)
        sizes_u = sizes[uvIds[:, 0]]
        sizes_v = sizes[uvIds[:, 1]]
        # union = size_up + size_dn - intersect
        unions = sizes_u + sizes_v - edge_lens
        # Union features
        topology_features[:, 2][z_mask] = unions[z_mask]
        topology_features_names.append("TopologyFeature_union")
        # IoU features
        topology_features[:, 3][z_mask] = edge_lens[z_mask] / unions[z_mask]
        topology_features_names.append("TopologyFeature_intersectionoverunion")

        # segment shape features
        seg_coordinates = extractor["RegionCenter"]
        len_bounds = np.zeros(rag.nodeNum)
        # TODO no loop ?! or CPP
        # iterate over the nodes, to get the boundary length of each node
        for n in rag.nodeIter():
            node_z = seg_coordinates[n.id][0]
            for arc in rag.incEdgeIter(n):
                edge = rag.edgeFromArc(arc)
                edge_c = rag.edgeCoordinates(edge)
                # only edges in the same slice!
                if edge_c[0, 0] == node_z:
                    len_bounds[n.id] += edge_lens[edge.id]
        # shape feature = Area / Circumference
        shape_feats_u = sizes_u / len_bounds[uvIds[:, 0]]
        shape_feats_v = sizes_v / len_bounds[uvIds[:, 1]]
        # combine w/ min, max, absdiff
        print(shape_feats_u[z_mask].shape)
        print(shape_feats_v[z_mask].shape)
        topology_features[:, 4][z_mask] = np.minimum(
            shape_feats_u[z_mask], shape_feats_v[z_mask])
        topology_features[:, 5][z_mask] = np.maximum(
            shape_feats_u[z_mask], shape_feats_v[z_mask])
        topology_features[:, 6][z_mask] = np.absolute(
            shape_feats_u[z_mask] - shape_feats_v[z_mask])
        topology_features_names.append("TopologyFeature_shapeSegment_min")
        topology_features_names.append("TopologyFeature_shapeSegment_max")
        topology_features_names.append("TopologyFeature_shapeSegment_absdiff")

        # edge shape features
        # this is too hacky, don't use it for now !
        # edge_bounds = np.zeros(rag.edgeNum)
        # adjacent_edges = self._adjacent_edges(seg_id)
        ## TODO no loop or CPP
        # for edge in rag.edgeIter():
        #    edge_coords = rag.edgeCoordinates(edge)
        #    edge_coords_up = np.ceil(edge_coords)
        #    #edge_coords_dn = np.floor(edge_coords)
        #    edge_z = edge_coords[0,2]
        #    for adj_edge_id in adjacent_edges[edge.id]:
        #        adj_coords = rag.edgeCoordinates(adj_edge_id)
        #        # only consider this edge, if it is in the same slice
        #        if adj_coords[0,2] == edge_z:
        #            # find the overlap and add it to the boundary
        #            #adj_coords_up = np.ceil(adj_coords)
        #            adj_coords_dn = np.floor(adj_coords)
        #            # overlaps (set magic...)
        #            ovlp0 = np.array(
        #                    [x for x in set(tuple(x) for x in edge_coords_up[:,:2])
        #                        & set(tuple(x) for x in adj_coords_dn[:,:2])] )
        #            #print edge_coords_up
        #            #print adj_coords_dn
        #            #print ovlp0
        #            #quit()
        #            #ovlp1 = np.array(
        #            #        [x for x in set(tuple(x) for x in edge_coords_dn[:,:2])
        #            #            & set(tuple(x) for x in adj_coords_up[:,:2])])
        #            #assert len(ovlp0) == len(ovlp1), str(len(ovlp0)) + " , " + str(len(ovlp1))
        #            edge_bounds[edge.id] += len(ovlp0)

        ## shape feature = Area / Circumference
        # topology_features[:,7][z_mask] = edge_lens[z_mask] / edge_bounds[z_mask]
        # topology_features_names.append("TopologyFeature_shapeEdge")

    topology_features = np.nan_to_num(topology_features)

    return topology_features, uvIds

def compute_region_features_(uv_ids, input_map, segmentation):
    """ Compute edge features from input map accumulated over segmentation
    and mapped to edges.

    Arguments:
        uv_ids [np.ndarray] - edge uv ids
        input_ [np.ndarray] - input data.
        segmentation [np.ndarray] - segmentation.
        n_threads [int] - number of threads used, set to cpu count by default. (default: None)
    """

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

## regular feature
def compute_regular_feature(watershed, raw, boundaries):
    if raw.ndim == 2:
        raw = raw[np.newaxis,...]
        watershed = watershed[np.newaxis,...]
        boundaries = boundaries[np.newaxis,...]
    rag = feats.compute_rag(watershed)
    feature1 = feats.compute_boundary_features_with_filters(rag, raw.astype('float32'), apply_2d=True)
    feature2 = feats.compute_boundary_features_with_filters(rag, boundaries.astype('float32'), apply_2d=True)
    # feature2 = compute_boundary_features(rag, boundaries)
    feature4, uvIds = topology_features( raw, watershed, use_2d_edges=True)
    feature3 = compute_region_features_(uvIds, raw.astype('float32'), watershed.astype('uint32'))
    ## resort
    idex=np.lexsort([uvIds[:,1], uvIds[:,0]])
    feature4 = feature4[idex, :]
    feature3 = feature3[idex, :]
    features = np.column_stack([feature1, feature2, feature3, feature4])
    return features, rag

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

def lifted_edge_label(rag, gt, uv_ids, n_threads=None):
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    node_labels = nrag.gridRagAccumulateLabels(rag, gt, n_threads)
    edge_labels = (node_labels[uv_ids[:, 0]] != node_labels[uv_ids[:, 1]]).astype('uint8')
    return edge_labels

## lifted feature
def compute_lifted_feature(watershed, raw, rag, pLocal, boundaries, n_threads=None, weighting_scheme="z"):
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    lifted_edge = lifted_edges_from_graph_neighborhood(rag, 2)
    feature3 = compute_region_features_(lifted_edge, raw.astype('float32'), watershed.astype('uint32'))
    feature4 = compute_lifted_feature_multicut(pLocal, rag, lifted_edge, boundaries, watershed, n_threads, weighting_scheme)
    features = np.column_stack([feature3, feature4])
    return features, lifted_edge

def single_mc(pLocal, edge_indications, edge_areas,
        rag, weighting_scheme, beta, weight):

    # copy the probabilities
    energies = mc.transform_probabilities_to_costs(pLocal, beta=beta)

    # weight the energies
    if weighting_scheme == "z":
        # print("Weighting Z edges")
        # z - edges are indicated with 0 !
        area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
        # we only weight the z edges !
        w = weight * edge_areas[edge_indications == 0] / area_z_max
        energies[edge_indications == 0] = np.multiply(w, energies[edge_indications == 0])

    elif weighting_scheme == "xyz":
        # print("Weighting xyz edges")
        # z - edges are indicated with 0 !
        area_z_max = float( np.max( edge_areas[edge_indications == 0] ) )
        len_xy_max = float( np.max( edge_areas[edge_indications == 1] ) )
        # weight the z edges !
        w_z = weight * edge_areas[edge_indications == 0] / area_z_max
        energies[edge_indications == 0] = np.multiply(w_z, energies[edge_indications == 0])
        # weight xy edges
        w_xy = weight * edge_areas[edge_indications == 1] / len_xy_max
        energies[edge_indications == 1] = np.multiply(w_xy, energies[edge_indications == 1])

    elif weighting_scheme == "all":
        # print("Weighting all edges")
        area_max = float( np.max( edge_areas ) )
        w = weight * edge_areas / area_max
        energies = np.multiply(w, energies)

    # get the energies (need to copy code here, because we can't use caching in threads)
    mc_node = mc.multicut_gaec(rag, energies)

    return mc_node

def compute_lifted_feature_multicut(pLocal, rag, lifted_edge, boundaries, watershed, n_threads, weighting_scheme):

    edge_areas = feats.compute_boundary_mean_and_length(rag, boundaries)[:, 1]
    edge_indications = feats.compute_z_edge_mask(rag, watershed).astype('uint8')
    # xy_edges = np.logical_not(z_edges)
    # edge_indications = [z_edges, xy_edges]

    uvIds = lifted_edge
    with futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        tasks = []
        for beta in (0.4, 0.45, 0.5, 0.55, 0.65):
            for w in (12, 16, 25):
                tasks.append( executor.submit( single_mc, pLocal, edge_indications, edge_areas,
        rag, weighting_scheme, beta, w) )

    mc_nodes = [future.result() for future in tasks]

    # map multicut result to lifted edges
    allFeat = [ ( mc_node[uvIds[:, 0]] !=  mc_node[uvIds[:, 1]] )[:,None] for mc_node in mc_nodes]

    mcStates = np.concatenate(allFeat, axis=1)
    stateSum = np.sum(mcStates,axis=1)
    return np.concatenate([mcStates, stateSum[:,None]],axis=1)

### load train data
path = 'path-to-raw_train.tif'
raw_train = imageio.volread(path)

path_pre = 'path-to-probabilities_train.tif'
boundaries_train = (imageio.volread(path_pre)/255).astype('float32')

path_gt = r'path-to-groundtruth.tif'
gt_train = imageio.volread(path_gt).astype('uint32')
gt_train = vigra.analysis.labelVolumeWithBackground(gt_train)
watershed_train = ws.stacked_watershed(boundaries_train, threshold=.3, sigma_seeds=2., alpha=.9, min_size=20)[0]
relabel_consecutive(watershed_train, block_shape=[5, 2048, 2048])

## test data
data_path = r'path-to-raw_test.tif'
boundaries_path = r'path-to-probabilities_test.tif'
save_path = r'save_path'
raw_test = imageio.volread(data_path)

boundaries_test = (imageio.volread(boundaries_path)/255).astype('float32')
watershed_test = ws.stacked_watershed(boundaries_test, threshold=.3, sigma_seeds=2., alpha=.9, min_size=20)[0]
relabel_consecutive(watershed_test, block_shape=[5, 2048, 2048])

# with napari.gui_qt():
#     viewer = napari.Viewer()
#     viewer.add_image(raw_train, name='raw')
#     viewer.add_image(boundaries_train, name='boundaries')
#     viewer.add_labels(watershed_train, name='watershed')
#     viewer.add_labels(gt_train, name='gt')
t0 = time.time()
features_total, rag = compute_regular_feature(watershed_train, raw_train, boundaries_train)
edge_total = elearn.compute_edge_labels(rag, gt_train)
rf = elearn.learn_edge_random_forest(features_total, edge_total, oob_score=True, n_estimators=500)
print('acc of xy:', rf.oob_score_)
pLocal = elearn.predict_edge_random_forest(rf, features_total)
lifted_features_total, lifted_edge = compute_lifted_feature(watershed_train, raw_train, rag, pLocal, boundaries_train)
lifted_edge_total = lifted_edge_label(rag, gt_train, lifted_edge)
rf_lifted = elearn.learn_edge_random_forest(lifted_features_total, lifted_edge_total, oob_score=True, n_estimators=500)
print('acc of lifted:', rf_lifted.oob_score_)
## test local edge
features_temp, rag = compute_regular_feature(watershed_test, raw_test, boundaries_test)
edge_pre = elearn.predict_edge_random_forest(rf, features_temp)
costs = mc.transform_probabilities_to_costs(edge_pre, beta=0.5)
## test lifted edge
lifted_features_test, lifted_edge_test = compute_lifted_feature(watershed_test, raw_test, rag, edge_pre, boundaries_test)
edge_pre_lifted = elearn.predict_edge_random_forest(rf_lifted, lifted_features_test)
costs_lifted = mc.transform_probabilities_to_costs(edge_pre_lifted, beta=0.5, weighting_exponent=2.)


## mc
node_labels = mc.multicut_fusion_moves(rag, costs)
segmentation_mc = feats.project_node_labels_to_pixels(rag, node_labels).astype('uint32')

## lmc
node_labels_lmc = lifted_multicut_fusion_moves(rag, costs, lifted_edge_test, costs_lifted)

segmentation_lmc = feats.project_node_labels_to_pixels(rag, node_labels_lmc)
print('all time is: ', time.time()-t0)
imageio.volwrite(save_path+'lifted_mulitcut_result.tif', segmentation_lmc.astype('uint32'))
print('end computer and save result')
