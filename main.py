import argparse
import os.path
import datetime

import torch
import numpy as np
from data_loader import load_data
from train import train, test

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-d', '--dataset', type=str, default='MOOCCube', help='which dataset to use (music, book, movie, restaurant)')
parser.add_argument('--data_dir_processed', type=str, default='data_processed', help='')

parser.add_argument('--n_worker', type=int, default=10, help='')
parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')
parser.add_argument('--course_size', type=int, default=20, help='')
parser.add_argument('--user_video_size', type=int, default=32, help='')
parser.add_argument('--user_triple_set_size', type=int, default=64, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=64, help='the number of triples in triple set of item')
parser.add_argument('--n_epoch', type=int, default=10, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')

parser.add_argument('--device', type=str, default='cuda:0', help='whether using gpu or cpu')
parser.add_argument('--use_random_seed', type=bool, default=True, help='whether using random seed or not')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='')
parser.add_argument('--state_dict', default=None, type=str)
parser.add_argument('--no_train', action="store_true", help="")

args = parser.parse_args()


def set_random_seed(np_seed, torch_seed):
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)


if __name__ == '__main__':
    if args.use_random_seed:
        set_random_seed(8848, 2233)
        # set_random_seed(2233, 8848)
    data_info = load_data(args)
    params_config = "{}-layer_{}-cs_{}-uts_{}-its_{}-uvs_{}-dim_{}".format(args.dataset, args.n_layer,
                                                                                            args.course_size,
                                                                                            args.user_triple_set_size,
                                                                                            args.item_triple_set_size,
                                                                                            args.user_video_size,
                                                                                            args.dim)
    # 为每次训练设置保存目录
    _checkpoint_dir = args.checkpoint_dir
    if not args.no_train:
        path = os.path.join(args.checkpoint_dir, args.dataset, params_config, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(path):
            os.makedirs(path)
        args.checkpoint_dir = path

        train(args, data_info)
    if args.state_dict is not None:
        args.checkpoint_dir = os.path.join(_checkpoint_dir, args.dataset, params_config, args.state_dict)
        args.state_dict_path = os.path.join(_checkpoint_dir, args.dataset, params_config, args.state_dict, "best.state_dict")
    else:
        args.state_dict_path = os.path.join(args.checkpoint_dir,  "best.state_dict")
    test(args, data_info)