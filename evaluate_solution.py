# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

import argparse
import os
import time
from subprocess import check_output


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'


def evaluate_solution(cap, net, res, verbose=False):
    start_time = time.time()
    if os.name == 'nt':
        exe_path = CURRENT_PATH + 'evaluator/evaluator_win64.exe'
    else:
        exe_path = CURRENT_PATH + 'evaluator/evaluator_linux.exe'
    if not os.path.isfile(exe_path):
        print('No exe file available: {}'.format(exe_path))
        exit()
    run_params = [exe_path, cap, net, res]
    res = check_output(run_params, timeout=36000).decode()
    if verbose is True:
        print(res)
    arr = res.split('\n')
    cost_wirelength = -1
    cost_via = -1
    cost_overflow = -1
    cost_total = -1
    for line in arr:
        line = line.strip()
        if 'wirelength cost' in line:
            cost_wirelength = float(line.split(' ')[-1])
        if 'via cost' in line:
            cost_via = float(line.split(' ')[-1])
        if 'overflow cost' in line:
            cost_overflow = float(line.split(' ')[-1])
        if 'total cost' in line:
            cost_total = float(line.split(' ')[-1])
    if verbose is True:
        print('Evaluation time: {:.2f} sec'.format(time.time() - start_time))
    return cost_wirelength, cost_via, cost_overflow, cost_total


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-cap", required=True, type=str, help="Location of cap file")
    parser.add_argument("-net", required=True, type=str, help="Location of net file")
    parser.add_argument("-output", required=True, type=str, help="Location of results file")
    args = parser.parse_args()

    print('Cap path: {}'.format(args.cap))
    print('Net path: {}'.format(args.net))
    print('Result path: {}'.format(args.output))
    cost_wirelength, cost_via, cost_overflow, cost_total = evaluate_solution(args.cap, args.net, args.output)
    print('Cost wirelength: {}'.format(cost_wirelength))
    print('Cost via: {}'.format(cost_via))
    print('Cost overflow: {}'.format(cost_overflow))
    print('Cost total: {}'.format(cost_total))
    print('Evaluation time: {:.2f} sec'.format(time.time() - start_time))


'''
-cap D:/Projects/2023_10_Global_Routing_ISPD_2024/input/nangate45/Simple_inputs/test.cap
-net D:/Projects/2023_10_Global_Routing_ISPD_2024/input/nangate45/Simple_inputs/test.net
-output ./result.txt
'''