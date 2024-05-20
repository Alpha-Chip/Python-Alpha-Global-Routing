# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

from utils import *


def get_ranges_for_net_simple(net_data):
    """
    Finds the coordinates of a rectangle that contains all the terminals for a node.
    It is necessary to extract the submatrix from the capacity matrix.
    """
    max_x = -1000000000
    min_x = 1000000000
    max_y = -1000000000
    min_y = 1000000000
    for points in net_data:
        z1, x1, y1 = points[0]
        if x1 > max_x:
            max_x = x1
        if x1 < min_x:
            min_x = x1
        if y1 > max_y:
            max_y = y1
        if y1 < min_y:
            min_y = y1

    w = max_x - min_x + 1
    h = max_y - min_y + 1
    r = (min_x, min_y, max_x, max_y, w, h)
    return r


def get_weight_from_value(val):
    res = (11 - val)
    return res


def find_solution_with_networkx(net):
    global matrix, data_cap, data_net

    layer_dir = data_cap['layerDirections']
    min_x, min_y, max_x, max_y, w, h = get_ranges_for_net_simple(data_net[net])
    submatrix = matrix[:, min_y:max_y + 1, min_x:max_x + 1].copy()

    id_to_index_map = dict()
    index_to_id_map = dict()
    G = nx.Graph()

    # First add all nodes
    node_id = 0
    for i in range(submatrix.shape[0]):  # z
        for j in range(submatrix.shape[1]):  # y
            for k in range(submatrix.shape[2]):  # x
                G.add_node(node_id + 1, terminal=False)
                id_to_index_map[(i, j, k)] = node_id + 1
                index_to_id_map[node_id + 1] = (i, j, k)
                node_id += 1

    # Add all GCell connections inside metal (first metal is not routable)
    for i in range(1, submatrix.shape[0]):
        if layer_dir[i] == 0:  # move only in x
            for j in range(submatrix.shape[1]):  # y
                for k in range(submatrix.shape[2] - 1):  # x
                    weight = 10 + get_weight_from_value(submatrix[i, j, k]) + get_weight_from_value(submatrix[i, j, k+1])
                    G.add_edge(id_to_index_map[(i, j, k)], id_to_index_map[(i, j, k + 1)], weight=weight)
        elif layer_dir[i] == 1:  # move only in y
            for j in range(submatrix.shape[1] - 1):  # y
                for k in range(submatrix.shape[2]):  # x
                    weight = 10 + get_weight_from_value(submatrix[i, j, k]) + get_weight_from_value(submatrix[i, j+1, k])
                    G.add_edge(id_to_index_map[(i, j, k)], id_to_index_map[(i, j + 1, k)], weight=weight)
        else:
            print('Error')
            exit()

    # Add all vias
    for i in range(submatrix.shape[0] - 1):  # z
        for j in range(submatrix.shape[1]):  # y
            for k in range(submatrix.shape[2]):  # x
                G.add_edge(id_to_index_map[(i, j, k)], id_to_index_map[(i + 1, j, k)], weight=40)

    # Create terminal nodes
    terminal_nodes_ret = []
    terminal_nodes = []
    for points in data_net[net]:
        z1, x1, y1 = points[0]
        terminal_nodes_ret.append((z1, y1 - min_y, x1 - min_x))
        terminal_nodes.append(id_to_index_map[(z1, y1 - min_y, x1 - min_x)])
        G.nodes[id_to_index_map[(z1, y1 - min_y, x1 - min_x)]]['terminal'] = True

    solution = nx.algorithms.approximation.steiner_tree(G, terminal_nodes, weight='weight', method="mehlhorn")

    nodes = []
    for v in solution.nodes():
        nodes.append(v)
    edges = []
    for u, v in solution.edges():
        edges.append((u, v))

    return nodes, edges, index_to_id_map, terminal_nodes_ret


def find_solution_for_net(net):
    global matrix, data_cap, data_net

    min_x, min_y, max_x, max_y, w, h = get_ranges_for_net_simple(data_net[net])
    submatrix = matrix[:, min_y:max_y + 1, min_x:max_x + 1].copy()
    nodes, edges, index_to_id_map, terminal_nodes = find_solution_with_networkx(net)
    update_capacity_matrix = np.zeros_like(submatrix)
    s1 = '{}\n(\n'.format(net)
    # Exotic case everything within one GCell
    if len(edges) == 0:
        tn = terminal_nodes[0]
        s1 += '{} {} {} {} {} {}\n'.format(min_x + tn[2], min_y + tn[1], tn[0], min_x + tn[2], min_y + tn[1], tn[0] + 1)
    else:
        for edge in edges:
            u = index_to_id_map[edge[0]]
            v = index_to_id_map[edge[1]]
            if v[0] < u[0] or v[1] < u[1] or v[2] < u[2]:
                u, v = v, u
            s1 += '{} {} {} {} {} {}\n'.format(min_x + u[2], min_y + u[1], u[0], min_x + v[2], min_y + v[1], v[0])
            update_capacity_matrix[u[0], u[1], u[2]] = 1
            update_capacity_matrix[v[0], v[1], v[2]] = 1
    s1 += ')\n'

    # Update capacity matrix
    matrix[:, min_y:max_y + 1, min_x:max_x + 1] -= update_capacity_matrix
    return s1


def route_circuit(
    output_file,
):
    import tqdm
    global matrix, data_cap, data_net

    matrix = data_cap['cap'].astype(np.float32)
    out = open(output_file, 'w')

    # Order of process nets
    data_proc = list(data_net.keys())

    # Sort by size of rectangle (we need to process large nets first)
    areas = []
    terminals = []
    dim_by_net = dict()
    for net in data_proc:
        min_x, min_y, max_x, max_y, w, h = get_ranges_for_net_simple(data_net[net])
        area = w*h
        areas.append(area)
        terminals.append(len(data_net[net]))
        dim_by_net[net] = (w, h)

    print('Sort nets by area...')
    areas = np.array(areas)
    # All nets which has pins in the single coordinate we process first
    areas[areas == 1] = 1000000000
    data_proc = sorted(zip(data_proc, areas), key=lambda x: x[1], reverse=True)
    data_proc = [i[0] for i in data_proc]

    bar = tqdm.tqdm(data_proc)
    for net in bar:
        bar.set_postfix({'net': net, 'size': dim_by_net[net]})
        s1 = find_solution_for_net(net)
        out.write(s1)

    out.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-cap", required=True, type=str, help="Location of cap file")
    parser.add_argument("-net", required=True, type=str, help="Location of net file")
    parser.add_argument("-output", required=True, type=str, help="Where to put the results")
    args = parser.parse_args()

    print('Read cap: {}'.format(args.cap))
    data_cap = read_cap(args.cap)
    print('Read net: {}'.format(args.net))
    data_net = read_net(args.net)

    print("Number of nets: {} Field size: {}x{}x{}".format(
        len(data_net),
        data_cap['nLayers'], data_cap['xSize'], data_cap['ySize'])
    )
    route_circuit(args.output)
    print('Overall time: {:.2f} sec'.format(time.time() - start_time))
