# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

from config import RouterConfig
from utils import *
from evaluate_solution import evaluate_solution

import argparse
import numpy as np
import time
import itertools
from multiprocessing import shared_memory, Lock, cpu_count, Pool, Manager
from subprocess import PIPE, Popen, check_output, DEVNULL
from skimage.measure import block_reduce
from hashlib import md5


CACHE_PATH = CURRENT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


def get_weight_from_value(val):
    res = (RouterConfig.MULTIPLY_COST * 15 - val)
    return res


def proc_with_ilya_binary(gr_string, verbose=False):
    """
    Transfers a line describing the graph to an Exe file with STP Solver. At the output we get the IDs of the nodes
    and the edges connecting them. The EXE does not open again, it hangs open. One EXE file per process.
    """
    global process_ilya

    process_ilya.stdin.write(gr_string)
    process_ilya.stdin.flush()

    nodes = set()
    edges = []
    while True:
        output = process_ilya.stdout.readline().strip()

        if len(output) == 0 or output == b'EOF':
            break

        arr = output.decode().split(" ")
        node1 = int(arr[1])
        node2 = int(arr[2])
        nodes |= {node1, node2}
        edges.append((node1, node2))

    process_ilya.stdin.flush()
    return nodes, edges


def proc_with_scip_jack(gr_string, net, verbose=False):
    if os.name == 'nt':
        exe_path = CURRENT_PATH + 'scip_jack/scipstp_win64.exe'
    else:
        exe_path = CURRENT_PATH + 'scip_jack/stp.linux.x86_64.gnu.opt.spx2'
    if not os.path.isfile(exe_path):
        print('No exe file available: {}'.format(exe_path))
        exit()

    # We need to create file to run SCIP-Jack binary on it
    net_name = md5(net.encode('utf-8')).hexdigest()
    out_file_store = CACHE_PATH + 'circ_{}_net_{}.gr'.format(
        circ_name,
        net_name
    )
    out = open(out_file_store, 'w')
    out.write(gr_string.decode())
    out.close()

    run_params = [exe_path, '-f', out_file_store, '-s', CURRENT_PATH + '/scip_jack/write.set']
    res = check_output(run_params, timeout=RouterConfig.TIMEOUT_FOR_SCIPJACK, stderr=DEVNULL, cwd=CACHE_PATH).decode()

    # os.chdir(cwd1)
    if verbose is True:
        print(res)
    arr = res.split('\n')

    # Parse answer
    nodes = []
    edges = []
    answer_path = CACHE_PATH + os.path.basename(out_file_store)[:-3] + '.stplog'

    in1 = open(answer_path)
    lines = in1.readlines()
    in1.close()

    total = 0
    num_vertices = -1
    for line in lines:
        line = line.strip()
        if 'Vertices' in line:
            num_vertices = int(line.split(" ")[-1])
            break
        total += 1

    lines = lines[total + 1:]

    total = 0
    for line in lines:
        line = line.strip()
        if 'Edges' in line:
            break
        _, id1 = line.split(" ")
        nodes.append(int(id1))
        total += 1

    if 0:
        if num_vertices != len(nodes):
            print('Some error with nodes: {} != {}'.format(num_vertices, len(nodes)))

    num_edges = int(lines[total].split(" ")[-1])
    lines = lines[total+1:]

    total = 0
    for line in lines:
        line = line.strip()
        if 'End' in line:
            break
        _, id1, id2 = line.split(" ")
        edges.append((int(id1), int(id2)))
        total += 1

    if num_edges != len(edges):
        print('Some error with edges: {} != {}'.format(num_edges, len(edges)))

    try:
        os.remove(answer_path)
    except Exception as e:
        pass

    try:
        os.remove(out_file_store)
    except Exception as e:
        pass

    # print(len(nodes), len(edges))
    return nodes, edges


def gen_string_in_gr_format_reduced(terminals, submatrix, layer_dir):
    """
    We provide the capacitance submatrix and the necessary terminals for the node that needs to be connected.
    The output is a graph in gr format - which can be given to Ilya’s STP code to obtain a Steiner tree.
    In this version of the function, all nodes and all edges are added.
    """
    out_string = []
    id_to_index_map = dict()
    index_to_id_map = dict()

    final_z_layer = submatrix.shape[0]

    # First add all nodes (first add all the nodes to the lists coordinates -> index and index -> coordinates)
    # This is necessary to restore the coordinates after receiving the solution from Ilya’s STP code,
    # because it will return the IDs.
    node_id = 0
    for i in range(final_z_layer):  # z
        for j in range(submatrix.shape[1]):  # y
            for k in range(submatrix.shape[2]):  # x
                id_to_index_map[(i, j, k)] = node_id + 1
                index_to_id_map[node_id + 1] = (i, j, k)
                node_id += 1


    edges_str = []

    # Count all edges
    # Add all GCell connections inside metal (first metal is not routable)
    # Тут быстро предрасчитывается матрица весов - это сделано для ускорения и может выглядеть сложно.
    # Можно посмотреть как на самом деле считается вес в старых версиях кода: router_experiment_v7.py
    # Там вес считается для каждого ребра отдельно что медленнее чем посчитать сразу для всей матрицы.

    if RouterConfig.TYPE_OF_EDGE_WEIGHT == 'sum':
        weights_matrix_horizontal = get_weight_from_value(submatrix[:, :, :-1]) + get_weight_from_value(submatrix[:, :, 1:])
        weights_matrix_vertical = get_weight_from_value(submatrix[:, :-1, :]) + get_weight_from_value(submatrix[:, 1:, :])
    else:
        weights_matrix_horizontal = np.maximum(get_weight_from_value(submatrix[:, :, :-1]), get_weight_from_value(submatrix[:, :, 1:]))
        weights_matrix_vertical = np.maximum(get_weight_from_value(submatrix[:, :-1, :]), get_weight_from_value(submatrix[:, 1:, :]))
    weights_matrix_horizontal += np.int16(RouterConfig.ADDITIONAL_EDGE_WEIGHT)
    weights_matrix_vertical += np.int16(RouterConfig.ADDITIONAL_EDGE_WEIGHT)
    for i in range(1, final_z_layer):
        if layer_dir[i] == 0:  # move only in x
            for j in range(submatrix.shape[1]):  # y
                for k in range(submatrix.shape[2] - 1):  # x
                    weight = weights_matrix_horizontal[i, j, k]
                    edges_str.append("E " + str(id_to_index_map[(i, j, k)]) + " " + str(id_to_index_map[(i, j, k + 1)]) + " " + str(weight) + "\n")
        elif layer_dir[i] == 1:  # move only in y
            for j in range(submatrix.shape[1] - 1):  # y
                for k in range(submatrix.shape[2]):  # x
                    weight = weights_matrix_vertical[i, j, k]
                    edges_str.append("E " + str(id_to_index_map[(i, j, k)]) + " " + str(id_to_index_map[(i, j + 1, k)]) + " " + str(weight) + "\n")
        else:
            print('Error')
            exit()

    # Add all vias
    for i in range(final_z_layer - 1):  # z
        for j in range(submatrix.shape[1]):  # y
            for k in range(submatrix.shape[2]):  # x
                # number_of_edges += 1
                edges_str.append("E " + str(id_to_index_map[(i, j, k)]) + " " + str(id_to_index_map[(i + 1, j, k)]) + " " + str(RouterConfig.VIA_COST) + "\n")

    out_string.append('SECTION Graph\n')
    out_string.append('Nodes ' + str(node_id) + '\n')
    out_string.append('Edges ' + str(len(edges_str)) + '\n')
    out_string.append(''.join(edges_str))
    out_string.append("END\n\n")
    out_string.append('SECTION Terminals\n')
    out_string.append('Terminals ' + str(len(terminals)) + '\n')

    # Create terminal nodes
    for point in terminals:
        z1, y1, x1 = point
        out_string.append("T " + str(id_to_index_map[(z1, y1, x1)]) + "\n")

    out_string.append("END\n\n")
    out_string.append("EOF\n")
    out_string = ''.join(out_string).encode()

    return out_string, index_to_id_map


def gen_string_in_gr_format_full(terminals, submatrix, mask, layer_dir):
    """
    Unlike the gen_string_in_gr_format_reduced function, here only nodes from the submatrix
    are added to the graph, which are marked 1 in the mask matrix. All others are ignored.
    For an edge to be added, both nodes must be marked 1.
    """
    out_string = []

    final_z_layer = submatrix.shape[0]

    # First add all nodes. We find them from the mask using one matrix operation.
    # We get 3 arrays with coordinates in locs. There are 3 arrays - because there are 3 dimensions.
    locs = np.where(mask > 0)

    # Next, we quickly number all the nodes that came into consideration
    a = np.arange(1, len(locs[0]) + 1)
    r1 = tuple(zip(locs[0], locs[1], locs[2]))
    index_to_id_map = dict(zip(a, r1))
    id_to_index_map = dict(zip(r1, a))
    node_id = a[-1]

    edges_str = []
    edge_weights = dict()

    # Count all edges
    # Add all GCell connections inside metal (first metal is not routable)
    # Similarly to gen_string_in_gr_format_reduced we obtain weight matrices for fast weight extraction in one operation
    if RouterConfig.TYPE_OF_EDGE_WEIGHT == 'sum':
        weights_matrix_horizontal = get_weight_from_value(submatrix[:, :, :-1]) + get_weight_from_value(submatrix[:, :, 1:])
        weights_matrix_vertical = get_weight_from_value(submatrix[:, :-1, :]) + get_weight_from_value(submatrix[:, 1:, :])
    else:
        weights_matrix_horizontal = np.maximum(get_weight_from_value(submatrix[:, :, :-1]), get_weight_from_value(submatrix[:, :, 1:]))
        weights_matrix_vertical = np.maximum(get_weight_from_value(submatrix[:, :-1, :]), get_weight_from_value(submatrix[:, 1:, :]))
    weights_matrix_horizontal += np.int16(RouterConfig.ADDITIONAL_EDGE_WEIGHT)
    weights_matrix_vertical += np.int16(RouterConfig.ADDITIONAL_EDGE_WEIGHT)
    ld = layer_dir.copy()
    # Here we make sure that layer 0 is not connected in any way - since you can’t make tracks in it
    ld[0] = 2
    # To speed up, we iterate over the nodes that came into consideration, because if you walk through the entire matrix, then it’s slow
    for i, j, k in id_to_index_map.keys():
        if ld[i] == 0:
            # We check that the second node for this edge is also included in the considered
            if (i, j, k + 1) in id_to_index_map:
                weight = weights_matrix_horizontal[i, j, k]
                edges_str.append("E " + str(id_to_index_map[(i, j, k)]) + " " + str(id_to_index_map[(i, j, k + 1)]) + " " + str(weight) + "\n")
        if ld[i] == 1:
            if (i, j + 1, k) in id_to_index_map:
                weight = weights_matrix_vertical[i, j, k]
                edges_str.append("E " + str(id_to_index_map[(i, j, k)]) + " " + str(id_to_index_map[(i, j + 1, k)]) + " " + str(weight) + "\n")
        if i < final_z_layer - 1:
            if (i + 1, j, k) in id_to_index_map:
                edges_str.append("E " + str(id_to_index_map[(i, j, k)]) + " " + str(id_to_index_map[(i + 1, j, k)]) + " " + str(RouterConfig.VIA_COST) + "\n")

    out_string.append('SECTION Graph\n')
    out_string.append('Nodes ' + str(node_id) + '\n')
    out_string.append('Edges ' + str(len(edges_str)) + '\n')
    out_string.append(''.join(edges_str))
    out_string.append("END\n\n")
    out_string.append('SECTION Terminals\n')
    out_string.append('Terminals ' + str(len(terminals)) + '\n')

    # Create terminal nodes
    for point in terminals:
        z1, y1, x1 = point
        out_string.append("T " + str(id_to_index_map[(z1, y1, x1)]) + "\n")

    out_string.append("END\n\n")
    out_string.append("EOF\n")
    out_string = ''.join(out_string).encode()

    return out_string, index_to_id_map, edge_weights


def find_solution_with_networkx(terminal_locations, submatrix, mask, layer_dir):
    id_to_index_map = dict()
    index_to_id_map = dict()
    G = nx.Graph()

    # First add all nodes
    node_id = 0
    for i in range(submatrix.shape[0]):  # z
        for j in range(submatrix.shape[1]):  # y
            for k in range(submatrix.shape[2]):  # x
                if mask[i, j, k] == 1:
                    G.add_node(node_id + 1, terminal=False)
                    id_to_index_map[(i, j, k)] = node_id + 1
                    index_to_id_map[node_id + 1] = (i, j, k)
                    node_id += 1

    # Add all GCell connections inside metal (first metal is not routable)
    for i in range(1, submatrix.shape[0]):
        if layer_dir[i] == 0:  # move only in x
            for j in range(submatrix.shape[1]):  # y
                for k in range(submatrix.shape[2] - 1):  # x
                    if mask[i, j, k] == 1 and mask[i, j, k + 1] == 1:
                        if RouterConfig.TYPE_OF_EDGE_WEIGHT == 'sum':
                            weight = get_weight_from_value(submatrix[i, j, k]) + get_weight_from_value(submatrix[i, j, k+1])
                        else:
                            weight = np.maximum(get_weight_from_value(submatrix[i, j, k]), get_weight_from_value(submatrix[i, j, k + 1]))
                        weight += np.int16(RouterConfig.ADDITIONAL_EDGE_WEIGHT)
                        G.add_edge(id_to_index_map[(i, j, k)], id_to_index_map[(i, j, k + 1)], weight=weight)
        elif layer_dir[i] == 1:  # move only in y
            for j in range(submatrix.shape[1] - 1):  # y
                for k in range(submatrix.shape[2]):  # x
                    if mask[i, j, k] == 1 and mask[i, j + 1, k] == 1:
                        if RouterConfig.TYPE_OF_EDGE_WEIGHT == 'sum':
                            weight = get_weight_from_value(submatrix[i, j, k]) + get_weight_from_value(submatrix[i, j+1, k])
                        else:
                            weight = np.maximum(get_weight_from_value(submatrix[i, j, k]) + get_weight_from_value(submatrix[i, j + 1, k]))
                        weight += np.int16(RouterConfig.ADDITIONAL_EDGE_WEIGHT)
                        G.add_edge(id_to_index_map[(i, j, k)], id_to_index_map[(i, j + 1, k)], weight=weight)
        else:
            print('Error')
            exit()

    # Add all vias
    for i in range(submatrix.shape[0] - 1):  # z
        for j in range(submatrix.shape[1]):  # y
            for k in range(submatrix.shape[2]):  # x
                if mask[i, j, k] == 1 and mask[i + 1, j, k] == 1:
                    G.add_edge(id_to_index_map[(i, j, k)], id_to_index_map[(i + 1, j, k)], weight=RouterConfig.VIA_COST)

    # Create terminal nodes
    terminal_nodes = []
    for points in terminal_locations:
        z1, y1, x1 = points
        terminal_nodes.append(id_to_index_map[(z1, y1, x1)])
        G.nodes[id_to_index_map[(z1, y1, x1)]]['terminal'] = True

    solution = nx.algorithms.approximation.steiner_tree(
        G,
        terminal_nodes,
        weight='weight',
        method="mehlhorn"
    )

    nodes = []
    for v in solution.nodes():
        nodes.append(v)
    edges = []
    for u, v in solution.edges():
        edges.append((u, v))

    return nodes, edges, index_to_id_map


def get_terminals(all_points, min_x, min_y):
    # Create terminal nodes. Recalculates the terminal coordinates relative to the edge of the extracted submatrix.
    terminal_nodes = []
    for points in all_points:
        z1, x1, y1 = points[0]
        terminal_nodes.append((z1, y1 - min_y, x1 - min_x))
    return terminal_nodes


def find_solution_for_net(net_data):
    """
   The main operating function finds the terminal connection for one node from the diagram.
   The function is called in parallel for the entire subset of nodes.
    """
    global matrix_shape, layer_dir

    net, data_net = net_data
    start_time = time.time()

    # We get the coordinates of a rectangle containing all terminals for a given net
    min_x, min_y, max_x, max_y, w, h = get_ranges_for_net(data_net[net], matrix_shape, gap=RouterConfig.GAP)

    # We store the matrix with capacities in shared memory. So that all processes work with the same matrix and change it
    existing_shm = shared_memory.SharedMemory(name=RouterConfig.SHARED_MEMORY_NAME)
    np_array = np.ndarray(matrix_shape, dtype=np.int16, buffer=existing_shm.buf)

    # Redundant case (for cases when all terminals are located at the same point, we output the solution immediately,
    # so as not to waste time)
    if w == h == 1:
        z1, x1, y1 = data_net[net][0][0]
        s1 = net + '\n(\n'
        s1 += str(x1) + ' ' + str(y1) + ' ' + str(z1) \
              + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(z1 + 1) + '\n'
        s1 += ')\n'

        lock.acquire()
        np_array[z1:z1+2, y1:y1+1, x1:x1+1] -= RouterConfig.MULTIPLY_COST
        lock.release()

        return s1, ('single cell net', )

    if RouterConfig.USE_PROCESSING_LOCK:
        existing_shm_lock = shared_memory.SharedMemory(name=RouterConfig.SHARED_MEMORY_NAME + '_lock')
        lock_array = np.ndarray(matrix_shape[1:], dtype=np.uint8, buffer=existing_shm_lock.buf)

        # We wait while our field is empty and ready to process
        while 1:
            lock.acquire()
            if lock_array[min_y:max_y + 1, min_x:max_x + 1].any():
                lock.release()
                time.sleep(RouterConfig.PROCESSING_LOCK_TIMEOUT)
                continue
            else:
                lock_array[min_y:max_y + 1, min_x:max_x + 1] = True
                lock.release()
                break

    """
         Extract the submatrix for the net. For large nodes we cannot use standard algorithm to solve - it is too slow.
         Therefore, we split the problem into two parts, first we reduce the matrix by N1xN2 times. Then we find 
         a solution on this reduced matrix. We draw the resulting tracks in a mask (a zero matrix, where the tracks 
         are marked with ones). We increase the mask by N1xN2 times. And then we find a Steiner tree for the graph 
         whose nodes are marked by one in this mask. This approach allows you to reduce computing resources 
         several times, but the quality of the solution also decreases.
    """
    submatrix_required_shape = (matrix_shape[0], max_y + 1 - min_y, max_x + 1 - min_x)
    failed_scip_jack = 0

    # We are looking for how many times to reduce the submatrix vertically and horizontally.
    # Now the value is calculated as the square root of the width and height. How to do it more efficient need to research.
    N1 = int(np.sqrt(submatrix_required_shape[1]))
    N2 = int(np.sqrt(submatrix_required_shape[2]))

    # We only do this trick for large matrices. If the matrix is small there is no point in running a double process.
    # The constant 30 is set by common sense. Research is required on what is best for speed and quality.
    if submatrix_required_shape[1] > RouterConfig.SKIP_DOUBLE_STAGE and submatrix_required_shape[2] > RouterConfig.SKIP_DOUBLE_STAGE:
        # We need to extract submatrix with shape divisible by N
        y_enlarge = (N1 - (max_y - min_y + 1) + N1 * ((max_y - min_y + 1) // N1)) % N1
        x_enlarge = (N2 - (max_x - min_x + 1) + N2 * ((max_x - min_x + 1) // N2)) % N2
        submatrix = np_array[:, min_y:max_y + 1 + y_enlarge, min_x:max_x + 1 + x_enlarge].copy()

        # Sometimes if the submatrix is on the edge, it is not possible to extract it as needed and you need
        # to add zeros to the desired size. The goal is for the matrix sizes to be divided into N1 and N2, respectively
        if submatrix.shape[1] % N1 != 0:
            add = N1 - submatrix.shape[1] % N1
            submatrix = np.pad(submatrix, ((0, 0), (0, add), (0, 0)), 'constant')
        if submatrix.shape[2] % N2 != 0:
            add = N2 - submatrix.shape[2] % N2
            submatrix = np.pad(submatrix, ((0, 0), (0, 0), (0, add)), 'constant')

        # We get terminals in submatrix coordinates
        terminal_locations = get_terminals(data_net[net], min_x, min_y)

        # Reduce the matrix submatrix by N1 x N2 times. The Z dimension remains the same.
        # We take the minimum value from the values for each N1xN2 block (you can also take the average,
        # but the solution is a little worse). Again need to investigate.
        submatrix_reduced = block_reduce(submatrix, block_size=(1, N1, N2), func=np.min)

        # The positions of the terminals in the reduced matrix have changed. We need to recalc them.
        terminal_reduced = []
        for z, y, x in terminal_locations:
            terminal_reduced.append((z, y // N1, x // N2))
            if y // N1 >= submatrix_reduced.shape[1]:
                print('Some error here 1!', y // N1, submatrix_reduced.shape[1])
            if x // N2 >= submatrix_reduced.shape[2]:
                print('Some error here 2!', x // N2, submatrix_reduced.shape[2])

        if RouterConfig.STP_ROUTER == 'networkx':
            mask = np.ones_like(submatrix_reduced)
            nodes, edges, index_to_id_map = find_solution_with_networkx(terminal_reduced, submatrix_reduced, mask, layer_dir)
        elif RouterConfig.STP_ROUTER == 'c_stp':
            # We obtain for the reduced matrix its representation in the form of a graph (a string with a GR description)
            out_string, index_to_id_map = gen_string_in_gr_format_reduced(terminal_reduced, submatrix_reduced, layer_dir)
            # We feed a line with a GR description as input to the STP-solver EXE and get an approximation of the Steiner tree
            nodes, edges = proc_with_ilya_binary(out_string)
        elif RouterConfig.STP_ROUTER == 'scip_jack':
            # We obtain for the reduced matrix its representation in the form of a graph (a string with a GR description)
            out_string, index_to_id_map = gen_string_in_gr_format_reduced(terminal_reduced, submatrix_reduced, layer_dir)
            try:
                nodes, edges = proc_with_scip_jack(out_string, net)
            except Exception as e:
                print(str(e))
                nodes, edges = proc_with_ilya_binary(out_string)
                failed_scip_jack += 1

        # Print path on copy of reduced submatrix. Next, we create a zero mask the size of the reduced capacity matrix.
        # And fill in the cells included in the solution from STP-solver EXE file with ones.
        mask = np.zeros_like(submatrix_reduced)
        for node in nodes:
            n = index_to_id_map[node]
            mask[n[0], n[1], n[2]] = 1

        # We increase the mask by (N1, N2) times. Up to the size of the initial submatrix.
        # Thus, the thickness of the path increased by (N1, N2) times.
        mask_enlarged = mask.repeat(N1, axis=1).repeat(N2, axis=2)

        # restore required shape of submatrix (because we pad it or take larger part for better resizing)
        # We increased the submatrix a little. Therefore, you need to take only the necessary part from it and from the mask.
        # submatrx and mask_enlarged are the same size.
        submatrix = submatrix[:, :submatrix_required_shape[1], :submatrix_required_shape[2]]
        mask_enlarged = mask_enlarged[:, :submatrix_required_shape[1], :submatrix_required_shape[2]]

        if RouterConfig.STP_ROUTER == 'networkx':
            # We get a graph only with nodes that are checked in the mask as needed
            nodes, edges, index_to_id_map = find_solution_with_networkx(terminal_locations, submatrix, mask_enlarged,
                                                                        layer_dir)
        elif RouterConfig.STP_ROUTER == 'c_stp':
            # We get a graph only with nodes that are checked in the mask as needed (a string with a GR description)
            out_string, index_to_id_map, edge_weights = gen_string_in_gr_format_full(terminal_locations, submatrix,
                                                                                     mask_enlarged, layer_dir)

            # We feed a line with a GR description as input to the STP-solver EXE and get an approximation
            # of the Steiner tree in real scale
            nodes, edges = proc_with_ilya_binary(out_string)
        elif RouterConfig.STP_ROUTER == 'scip_jack':
            # We get a graph only with nodes that are checked in the mask as needed (a string with a GR description)
            out_string, index_to_id_map, edge_weights = gen_string_in_gr_format_full(terminal_locations, submatrix,
                                                                                     mask_enlarged, layer_dir)
            # We feed a line with a GR description as input to the STP-solver EXE and get an approximation
            # of the Steiner tree in real scale
            try:
                nodes, edges = proc_with_scip_jack(out_string, net)
            except Exception as e:
                print(str(e))
                nodes, edges = proc_with_ilya_binary(out_string)
                failed_scip_jack += 1
    else:
        # Default solution on full matrix (for small matrices). If the matrix is small, we immediately find a
        # solution on a full scale in one pass
        submatrix = np_array[:, min_y:max_y + 1, min_x:max_x + 1].copy()
        terminal_locations = get_terminals(data_net[net], min_x, min_y)

        if RouterConfig.STP_ROUTER == 'networkx':
            mask = np.ones_like(submatrix)
            nodes, edges, index_to_id_map = find_solution_with_networkx(terminal_locations, submatrix, mask, layer_dir)
        elif RouterConfig.STP_ROUTER == 'c_stp':
            out_string, index_to_id_map = gen_string_in_gr_format_reduced(terminal_locations, submatrix, layer_dir)
            nodes, edges = proc_with_ilya_binary(out_string)
        elif RouterConfig.STP_ROUTER == 'scip_jack':
            out_string, index_to_id_map = gen_string_in_gr_format_reduced(terminal_locations, submatrix, layer_dir)
            try:
                nodes, edges = proc_with_scip_jack(out_string, net)
            except Exception as e:
                print(str(e))
                nodes, edges = proc_with_ilya_binary(out_string)
                failed_scip_jack += 1


    '''
    Once we have found the solution, we need to update the large capacity matrix. 
    That is, subtract the capacity of the resulting path for all coordinates where it passes. 
    To do this, create a zero matrix, go through all the coordinates of the resulting 
    Steiner tree and put the value 1 * MULTIPLY_COST there. At the same time generate a 
    line with a solution for the output file
    '''

    update_capacity_matrix = np.zeros_like(submatrix)
    s1 = net + '\n(\n'
    # Exotic case everything within one GCell
    if len(edges) == 0:
        tn = terminal_locations[0]
        s1 += str(min_x + tn[2]) \
              + ' ' + str(min_y + tn[1]) \
              + ' ' + str(tn[0]) \
              + ' ' + str(min_x + tn[2]) \
              + ' ' + str(min_y + tn[1]) \
              + ' ' + str(tn[0] + 1) + '\n'
    else:
        for edge in edges:
            u = index_to_id_map[edge[0]]
            v = index_to_id_map[edge[1]]
            if v[0] < u[0] or v[1] < u[1] or v[2] < u[2]:
                u, v = v, u
            s1 += str(min_x + u[2]) \
                  + ' ' + str(min_y + u[1]) \
                  + ' ' + str(u[0]) \
                  + ' ' + str(min_x + v[2]) \
                  + ' ' + str(min_y + v[1]) \
                  + ' ' + str(v[0]) + '\n'

            update_capacity_matrix[u[0], u[1], u[2]] = RouterConfig.MULTIPLY_COST
            update_capacity_matrix[v[0], v[1], v[2]] = RouterConfig.MULTIPLY_COST
    s1 += ')\n'

    '''
    When we have filled the capacity reduction matrix (update_capacity_matrix) for a given node, 
    we lock our main matrix in shared memory so that other processes cannot change it and subtract 
    update_capacity_matrix from it. Next, we unblock it from recording.
    '''

    lock.acquire()
    np_array[:, min_y:max_y + 1, min_x:max_x + 1] -= update_capacity_matrix
    if RouterConfig.USE_PROCESSING_LOCK:
        lock_array[min_y:max_y + 1, min_x:max_x + 1] = False
    lock.release()
    existing_shm.close()
    if RouterConfig.USE_PROCESSING_LOCK:
        existing_shm_lock.close()
    if RouterConfig.STP_ROUTER == 'scip_jack':
        return s1, (failed_scip_jack, N1, N2, submatrix.shape)
    else:
        return s1, (N1, N2, submatrix.shape)


def init_worker(lock1, matrix1_shape1, layer_dir1, circ_name1):
    """
    This function is called when a process is initialized in a multiprocessor program.
    Here, for small variables that do not change, we will make them global.
    We launch the EXE file with STP solver and STP then waits for data from his parent process.
    """
    global lock, matrix_shape, layer_dir, circ_name, process_ilya
    lock = lock1
    matrix_shape = matrix1_shape1
    layer_dir = layer_dir1
    circ_name = circ_name1

    if os.name == 'nt':
        exe_path = CURRENT_PATH + 'stp_solver/bin/steiner_tree_problem_solver_win64.exe'
    else:
        exe_path = CURRENT_PATH + 'stp_solver/bin/steiner_tree_problem_solver_linux.exe'
    if not os.path.isfile(exe_path):
        print('No exe file available: {}'.format(exe_path))
        exit()

    process_ilya = Popen(
        [exe_path] + ["--stream"] + ["yes"],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE
    )

def create_shared_block(matrix, shared_name=RouterConfig.SHARED_MEMORY_NAME):
    """
    We create a capacity matrix in shared memory. Thus, each process will be able to both read from it and
    modify this matrix.
    """
    try:
        shm = shared_memory.SharedMemory(name=shared_name, create=True, size=matrix.nbytes)
    except Exception as e:
        # print('Error creating new shared memory... Use the same: {}'.format(e))
        shm = shared_memory.SharedMemory(name=shared_name, create=False, size=matrix.nbytes)
    # # Now create a NumPy array backed by shared memory
    np_array = np.ndarray(matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
    np_array[:] = matrix[:]  # Copy the original data into shared memory
    print('Created shared memory: {} Size: {}'.format(shared_name, shm.size))
    return shm, np_array


def route_circuit(
    circ_name,
    output_file,
    data_cap,
    data_net,
    num_procs,
):
    import tqdm
    matrix = np.round(RouterConfig.MULTIPLY_COST * data_cap['cap']).astype(np.int16)
    layer_dir = data_cap['layerDirections']
    out = open(output_file, 'w')

    # Order of process nets
    if RouterConfig.SORT_NETS_BY == 'area':
        data_proc = sort_nets_by_area(data_net, matrix.shape)
    elif RouterConfig.SORT_NETS_BY == 'min_intersection':
        data_proc = sort_nets_minimize_intersection(data_net, matrix.shape)
    elif RouterConfig.SORT_NETS_BY == 'random':
        data_proc = sort_nets_random(data_net, matrix.shape)

    cpus = cpu_count()
    use_cpus = num_procs
    if num_procs == -1:
        use_cpus = cpus // 2
    print('Number of CPUs: {} Use CPUs: {}'.format(cpus, use_cpus))

    # We place the capacity matrix in shared memory for all processes
    shm, np_array = create_shared_block(matrix)
    if RouterConfig.USE_PROCESSING_LOCK:
        matrix_lock = np.zeros(data_cap['cap'].shape[1:], dtype=np.uint8)
        shm_lock, lock_array = create_shared_block(matrix_lock, RouterConfig.SHARED_MEMORY_NAME + '_lock')

    # Do batches. In general, if the cycle is very long, then multiprocessing fails (maybe Windows issue).
    # Therefore, all nets were divided into batches of 250K. In general, it is not needed and all nets can
    # be counted in one pass
    batch_size = RouterConfig.BATCH_SIZE
    # Count some statistics
    for start_point in range(0, len(data_proc), batch_size):
        batch = data_proc[start_point:start_point + batch_size]
        print('Start batch from {} to {}'.format(start_point, start_point + len(batch)))

        # Try to avoid copy of big data in each process. Store it one time and share between processes.
        # It's not changed in process
        data_net_part = {key: data_net[key] for key in batch}
        data_net_part = Manager().dict(data_net_part)

        # Create all processes
        lock = Lock()
        pool = Pool(
            processes=use_cpus,
            initializer=init_worker,
            initargs=(lock, matrix.shape, layer_dir, circ_name)
        )

        with tqdm.tqdm(total=len(batch)) as pbar:
            # We start processing nets in multiprocessing. During processing we display some statistics
            for s1, stat in pool.imap(find_solution_for_net, zip(batch, itertools.repeat(data_net_part))):
                out.write(s1)
                pbar.set_postfix({'stat': stat})
                pbar.update()

    shm.unlink()
    if RouterConfig.USE_PROCESSING_LOCK:
        shm_lock.unlink()
    out.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-cap", required=True, type=str, help="Location of cap file")
    parser.add_argument("-net", required=True, type=str, help="Location of net file")
    parser.add_argument("-output", required=True, type=str, help="Where to put the results")
    parser.add_argument("-procs", type=int, default=-1, help="Number of processes to use. By default half of number of CPUs.")
    parser.add_argument("-metrics", action='store_true', help="If provided then metrics are calculated in the end")
    args = parser.parse_args()

    print('Read cap: {}'.format(args.cap))
    data_cap = read_cap(args.cap, verbose=True)
    print('Read net: {}'.format(args.net))
    data_net = read_net(args.net, verbose=True)
    output_file = args.output
    circ_name = 'userdefined'

    print("Go for: {} Number of nets: {} Field size: {}x{}x{}".format(
        circ_name,
        len(data_net),
        data_cap['nLayers'], data_cap['xSize'], data_cap['ySize'])
    )
    print("Config parameters")
    attrs = vars(RouterConfig)
    for attr in attrs:
        if '__' not in attr:
            print("{}: {}".format(attr, attrs[attr]))

    route_circuit(
        circ_name,
        output_file,
        data_cap,
        data_net,
        args.procs,
    )

    # Calculating the quality metrics in C-code
    if args.metrics is not None:
        cost_wirelength, cost_via, cost_overflow, cost_total = evaluate_solution(args.cap, args.net, output_file, verbose=True)
    print('Overall time: {:.2f} sec'.format(time.time() - start_time))
