# coding: utf-8
__author__ = 'Roman Solovyev: https://github.com/ZFTurbo'

from utils import *
import cv2
import argparse
import numpy as np
import time


CACHE_PATH = CURRENT_PATH + 'cache/'
if not os.path.isdir(CACHE_PATH):
    os.mkdir(CACHE_PATH)


def create_video(image_list, out_file, fps, codec):
    height, width = image_list[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'DIB ')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # fourcc = -1
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height), True)

    for i in range(image_list.shape[0]):
        im = image_list[i]
        if len(im.shape) == 2:
            im = np.stack((im, im, im), axis=2)
        video.write(im.astype(np.uint8))
    cv2.destroyAllWindows()
    video.release()


def read_pr_output(pr_filename, verbose=False):
    start_time = time.time()
    res = dict()
    in1 = open(pr_filename, 'r')

    total = 0
    lines = in1.readlines()
    in1.close()
    if verbose:
        print('Reading pr output file in memory finished... Processing...')

    progressbar = tqdm.tqdm(total=len(lines))
    while 1:
        if total >= len(lines):
            break
        name = lines[total].strip()
        total += 1
        progressbar.update(1)
        if name == '':
            break
        points = []
        while 1:
            line = lines[total].strip()
            total += 1
            progressbar.update(1)
            if line == '(':
                continue
            if line == ')':
                break
            r = line.strip().split(' ')
            r = [int(f) for f in r]
            points.append(r)
        if len(points) == 0:
            print('Zero points for {}...'.format(name))
            exit()

        res[name] = tuple(points)

    progressbar.close()
    if verbose:
        print('Reading pr output time: {:.2f} sec'.format(time.time() - start_time))
    return res


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-cap", required=True, type=str, help="Location of cap file")
    parser.add_argument("-net", required=True, type=str, help="Location of net file")
    parser.add_argument("-pr", required=True, type=str, help="Location of PR output file")
    parser.add_argument("-mp4", required=True, type=str, help="Name of MP4 to store results")
    args = parser.parse_args()

    print('Read output: {}'.format(args.pr))
    data_out = read_pr_output(args.pr, verbose=True)
    print('Read cap: {}'.format(args.cap))
    data_cap = read_cap(args.cap, verbose=True)
    print('Read net: {}'.format(args.net))
    data_net = read_net(args.net, verbose=True)

    matrix = np.round(data_cap['cap']).astype(np.int32)
    via_matrix = np.zeros(data_cap['cap'].shape, dtype=np.int16)

    max_vals = []
    for i in range(matrix.shape[0]):
        max_vals.append(matrix[i].max())

    print(matrix.shape, matrix.min(), matrix.max(), matrix.mean())
    total = 0
    all_frames = []
    for net in tqdm.tqdm(data_out):
        for r in data_out[net]:
            x1, y1, z1, x2, y2, z2 = r
            if z1 == z2:
                if y1 != y2:
                    matrix[z1:z1 + 1, y1:y2, x1:x1 + 1] -= 1
                else:
                    matrix[z1:z1 + 1, y1:y1 + 1, x1:x2] -= 1
        total += 1
        if total % 400 == 0:
            rs = 255 * matrix / np.expand_dims(np.expand_dims(np.array(max_vals).astype(np.float32), axis=-1), axis=-1)
            rs = np.stack([rs, rs, rs], axis=-1)
            rs[..., 0][rs[..., 0] < 0] = 0
            rs[..., 1][rs[..., 1] < 0] = 0
            rs[..., 2][rs[..., 2] < 0] = 255
            all_frames.append(rs.astype(np.uint8))

    all_frames = np.array(all_frames).astype(np.uint8)
    print(all_frames.shape)
    for i in range(all_frames.shape[1]):
        create_video(all_frames[:, i], args.mp4[:-4] + '_{}.mp4'.format(i), 30, 'H264')
    print('Overall time: {:.2f} sec'.format(time.time() - start_time))

