# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

import tqdm
import argparse
import os
import numpy as np
import networkx as nx
import time

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'

def read_cap(path, verbose=False):
    start_time = time.time()
    res = dict()
    in1 = open(path, 'r')

    line = in1.readline()
    nLayers, xSize, ySize = line.strip().split(" ")
    nLayers = int(nLayers)
    xSize = int(xSize)
    ySize = int(ySize)

    line = in1.readline()
    arr = line.strip().split(" ")
    unit_length_wire_cost = float(arr[0])
    unit_via_cost = float(arr[1])
    unit_length_short_costs = []
    for i in range(len(arr) - 2):
        unit_length_short_costs.append(float(arr[i+2]))

    line = in1.readline()
    arr = line.strip().split(" ")
    horizontal_GCell_edge_lengths = []
    for i in range(len(arr)):
        horizontal_GCell_edge_lengths.append(int(arr[i]))

    line = in1.readline()
    arr = line.strip().split(" ")
    vertical_GCell_edge_lengths = []
    for i in range(len(arr)):
        vertical_GCell_edge_lengths.append(int(arr[i]))

    layerNames = []
    layerDirections = []
    layerMinLengths = []
    cap = np.zeros((nLayers, ySize, xSize), dtype=np.float32)
    for i in range(nLayers):
        line = in1.readline()
        name, direction, min_length = line.strip().split(" ")
        layerNames.append(name)
        layerDirections.append(int(direction))
        layerMinLengths.append(int(min_length))
        for j in range(ySize):
            line = in1.readline()
            arr = line.strip().split(" ")
            arr = np.array(arr, dtype=np.float32)
            cap[i, j, :] = arr

    res['nLayers'] = nLayers
    res['xSize'] = xSize
    res['ySize'] = ySize
    res['unit_length_wire_cost'] = unit_length_wire_cost
    res['unit_via_cost'] = unit_via_cost
    res['unit_length_short_costs'] = unit_length_short_costs
    res['horizontal_GCell_edge_lengths'] = horizontal_GCell_edge_lengths
    res['vertical_GCell_edge_lengths'] = vertical_GCell_edge_lengths
    res['layerNames'] = layerNames
    res['layerDirections'] = layerDirections
    res['layerMinLengths'] = layerMinLengths

    # direction dict
    dd = dict()
    for i in range(len(layerDirections)):
        dd[res['layerNames'][i]] = res['layerDirections'][i]

    # metal level
    ml = dict()
    for i in range(len(layerDirections)):
        ml[res['layerNames'][i]] = i

    res['cap'] = cap
    res['dir'] = dd
    res['level'] = ml

    if verbose:
        print('Reading caps time: {:.2f} sec'.format(time.time() - start_time))
    in1.close()
    return res


def read_net(path, verbose=False):
    start_time = time.time()
    res = dict()
    in1 = open(path, 'r')

    total = 0
    lines = in1.readlines()
    in1.close()
    if verbose:
        print('Reading net file in memory finished... Processing...')

    progressbar = tqdm.tqdm(total=len(lines))
    while 1:
        if total >= len(lines):
            break
        name = lines[total].strip(); total += 1; progressbar.update(1)
        if name == '':
            break
        points = []
        while 1:
            line = lines[total].strip(); total += 1; progressbar.update(1)
            if line == '(':
                continue
            if line == ')':
                break
            r = eval(line)
            points.append(r)
        if len(points) == 0:
            print('Zero points for {}...'.format(name))
            exit()

        res[name] = tuple(points)

    progressbar.close()
    if verbose:
        print('Reading nets time: {:.2f} sec'.format(time.time() - start_time))
    return res


def get_ranges_for_net(net_data, matrix_shape, gap=0):
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

    # Increase the size of rectangle (for more flexibility of algorithm). It can be slower with big GAP.
    if gap > 0:
        min_x = max(0, min_x - gap)
        min_y = max(0, min_y - gap)
        max_x = min(matrix_shape[2] - 1, max_x + gap)
        max_y = min(matrix_shape[1] - 1, max_y + gap)

    w = max_x - min_x + 1
    h = max_y - min_y + 1
    r = (min_x, min_y, max_x, max_y, w, h)
    return r

def sort_nets_by_area(data_net, matrix_shape):
    # Order of process nets
    data_proc = list(data_net.keys())

    # Sort by size of rectangle (we need to process large nets first)
    areas = []
    terminals = []
    dim_by_net = dict()
    for net in data_proc:
        min_x, min_y, max_x, max_y, w, h = get_ranges_for_net(data_net[net], matrix_shape)
        area = w * h
        areas.append(area)
        terminals.append(len(data_net[net]))
        dim_by_net[net] = (w, h)

    print('Sort nets by area...')
    areas = np.array(areas)
    # Nets with all terminals at the same coordinate are processed first
    areas[areas == 1] = 1000000000
    data_proc = sorted(zip(data_proc, areas), key=lambda x: x[1], reverse=True)
    data_proc = [i[0] for i in data_proc]
    return data_proc


def sort_nets_random(data_net, matrix_shape):
    data_proc = list(data_net.keys())

    # Get data for all nets
    r = dict()
    areas = []
    for net in data_proc:
        min_x, min_y, max_x, max_y, w, h = get_ranges_for_net(data_net[net], matrix_shape)
        area = w * h
        areas.append(area)
        r[net] = (min_x, min_y, max_x + 1, max_y + 1, area, len(data_net[net]))

    areas = np.array(areas)
    data_proc = np.array(data_proc)

    # Include single cell nets first
    small_nets = data_proc[areas == 1]
    other_nets = data_proc[areas != 1]
    np.random.shuffle(other_nets)
    order = list(small_nets) + list(other_nets)

    return order


def sort_nets_minimize_intersection(data_net, matrix_shape):
    start_time = time.time()
    # Order of process nets
    data_proc = list(data_net.keys())

    # Get data for all nets
    r = dict()
    areas = []
    for net in data_proc:
        min_x, min_y, max_x, max_y, w, h = get_ranges_for_net(data_net[net], matrix_shape)
        area = w * h
        areas.append(area)
        r[net] = (min_x, min_y, max_x + 1, max_y + 1, area, len(data_net[net]))

    # Here will be final order of nets
    order = []

    print('Sort nets to minimize intersection during processing...')
    areas = np.array(areas)
    data_proc = np.array(data_proc)

    # Include single cell nets first
    small_nets = data_proc[areas == 1]
    order += list(small_nets)

    # Exclude single cell nets from further considering
    data_proc = data_proc[areas != 1]
    areas = areas[areas != 1]

    # Nets with all terminals at the same coordinate are processed first
    data_proc = sorted(zip(data_proc, areas), key=lambda x: x[1], reverse=True)
    data_proc = [i[0] for i in data_proc]

    while 1:
        current_nets = []
        m = np.zeros(matrix_shape[1:], dtype=np.bool_)
        for i in range(len(data_proc)):
            net = data_proc[i]
            x1, y1, x2, y2, area, terminals = r[net]
            if not m[y1:y2, x1:x2].any():
                current_nets.append(net)
                m[y1:y2, x1:x2] = True
                if m.sum() > 0.95 * m.shape[0] * m.shape[1]:
                    print("Break: {}/{} Density: {}".format(i, len(data_proc), m.sum() / (m.shape[0] * m.shape[1])))
                    break
        order += current_nets
        filter_set = set(current_nets)
        if 0:
            # Save order
            data_proc = [x for x in data_proc if x not in filter_set]
        else:
            # Unordered
            data_proc = list(set(data_proc) - filter_set)
        print(len(current_nets), len(data_proc))
        if len(data_proc) <= 0:
            break

    print('Sorting nets time: {:.2f} sec'.format(time.time() - start_time))
    return order[::-1]
