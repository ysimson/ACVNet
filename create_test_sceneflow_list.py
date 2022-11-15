# from __future__ import print_function, division
import argparse
import os
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')

    parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
    parser.add_argument('--testlist', default='./filenames/sceneflow_test_sample.txt', help='testing list')

    # parse arguments, set seeds
    args = parser.parse_args()

    #
    root = args.datapath
    disparity_list = glob.glob(os.path.join(root, "**/disparity/*.pfm"), recursive=True)
    left_rgb_list = glob.glob(os.path.join(root, "**/RGB_cleanpass/left/*.png"), recursive=True)

    right_rgb_list = []
    for left_fn in left_rgb_list:
        right_folder = os.path.abspath(os.path.join(os.path.dirname(left_fn), "../right"))
        right_fn = os.path.join(right_folder, os.path.basename(left_fn))
        right_rgb_list.append(right_fn)

    disparity_list = []
    for left_fn in left_rgb_list:
        disparity_folder = os.path.abspath(os.path.join(os.path.dirname(left_fn), "../../disparity"))
        filename = os.path.basename(left_fn)

        disparity_fn = os.path.join(disparity_folder, os.path.splitext(filename)[0] + ".pfm")
        disparity_list.append(disparity_fn)

    lines = []
    for left_fn, right_fn, disparity_fn in zip(left_rgb_list, right_rgb_list, disparity_list):
        datasets_str_len = len(root) + 1
        line = " ".join([left_fn[datasets_str_len:], right_fn[datasets_str_len:], disparity_fn[datasets_str_len:], "\n"])
        lines.append(line)

    with open(args.testlist, "w") as f:
        f.writelines(lines)
    print("Done")
