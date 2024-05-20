# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

class RouterConfig:
    STP_ROUTER = 'c_stp' # 'networkx', 'c_stp', 'scip_jack'
    SORT_NETS_BY = 'random'  # 'random', 'area', 'min_intersection'
    SKIP_DOUBLE_STAGE = 10 # Skip double stage algorithm if rectangle side is less than this value. You can put it very large to use single stage algorithm for all nets
    MULTIPLY_COST = 5 # We need this because we do rounding of capacity and store all in int16 matrix
    VIA_COST = 10.0 * MULTIPLY_COST # How much cost for via
    GAP = 0 # Size of border about set of terminals - larger give more flexibility to find lower cost path, but complexity of problem is increased (doesn't really give better result for > 0)
    TYPE_OF_EDGE_WEIGHT = 'sum' # 'sum' or 'min'
    ADDITIONAL_EDGE_WEIGHT = 0.0 * MULTIPLY_COST # initial weight of edge before adding vertex weights
    BATCH_SIZE = 200000 # Set as max as possible (only needed to solve strange problem with "out of memory" in windows)
    USE_PROCESSING_LOCK = True # If true processing regions will be locked for other processes
    PROCESSING_LOCK_TIMEOUT = 0.0 # How much time to wait to try again to process region
    TIMEOUT_FOR_SCIPJACK = 30
    SHARED_MEMORY_NAME = 'capacity_matrix_router' # can be any name (if you run two routers at the same time names must be different!)
    DEBUG = False # Do some printing if True
