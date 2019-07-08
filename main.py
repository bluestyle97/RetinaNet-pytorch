import argparse
from solver import Solver

from datasets.coco import CocoDataLoader
from utils.misc import create_dirs


def parse():
    parser = argparse.ArgumentParser(description='Object Detection with RetinaNet')
    subparsers = parser.add_subparsers(help='sub command', dest='command')
    subparsers.required = True

    # train parameters
    parser_train = subparsers.add_parser('train', help='train retinanet')
    parser_train.add_argument('--checkpoint', action='store', type=int, help='checkpoint to resume from')
    parser_train.add_argument('--backbone', type=str, default='fpn50', help='backbone of retinanet')
    parser_train.add_argument('--dataset', type=str, default='coco', help='dataset type')
    parser_train.add_argument('--data_root', type=str, default='/root/datasets/COCO', help='root path of dataset')
    parser_train.add_argument('--num_classes', type=int, default=80, help='number of object categories')
    parser_train.add_argument('--num_features', type=int, default=256, help='number of features in two subnets')
    parser_train.add_argument('--cuda', action='store_true', help='use cuda or not')
    parser_train.add_argument('--resize', type=int, default=800, help='resize images to given size')
    parser_train.add_argument('--max_size', type=int, default=1333, help='maximum image size')
    parser_train.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser_train.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser_train.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser_train.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser_train.add_argument('--milestones', type=int, nargs='*', default=[60000, 80000], help='epochs after which learning rate decays')
    parser_train.add_argument('--focal_alpha', type=float, default=0.25, help='alpha value of focal loss')
    parser_train.add_argument('--focal_gamma', type=float, default=2.0, help='gamma value of focal loss')
    parser_train.add_argument('--smoothl1_beta', type=float, default=0.11, help='beta value of smoothl1 loss')
    parser_train.add_argument('--max_iters', type=int, default=90000, help='maximum epochs')
    parser_train.add_argument('--summary_step', type=int, default=100, help='epochs per summary operation')
    parser_train.add_argument('--save_step', type=int, default=10000, help='epochs per saving operation')
    parser_train.add_argument('--val_step', type=int, default=1000, help='epochs per validation operation')
    parser_train.add_argument('--summary_dir', type=str, default='experiments/summaries', help='directory to summarize in')
    parser_train.add_argument('--checkpoint_dir', type=str, default='experiments/checkpoints', help='directory to save checkpoints in')
    parser_train.add_argument('--log_dir', type=str, default='experiments/logs', help='directory to save logs in')
    parser_train.add_argument('--dec_threshold', type=float, default=0.05, help='threshold when decoding output into boxes')
    parser_train.add_argument('--nms_threshold', type=float, default=0.5, help='threshold when performing non-maximum suppression')
    parser_train.add_argument('--topn', type=int, default=1000, help='top n boxes when decoding')
    parser_train.add_argument('--ndetections', type=int, default=100, help='maximum boxes number after nms')

    parser_infer = subparsers.add_parser('infer', help='run inference')
    parser_infer.add_argument('--img_path', type=str, help='path of input image')
    parser_infer.add_argument('--checkpoint', action='store', type=int, help='checkpoint to resume from')
    parser_infer.add_argument('--backbone', type=str, default='fpn50', help='backbone of retinanet')
    parser_infer.add_argument('--num_classes', type=int, default=80, help='number of object categories')
    parser_infer.add_argument('--num_features', type=int, default=256, help='number of features in two subnets')
    parser_infer.add_argument('--cuda', action='store_true', help='use cuda or not')
    parser_infer.add_argument('--resize', type=int, default=800, help='resize images to given size')
    parser_infer.add_argument('--max_size', type=int, default=1333, help='maximum image size')
    parser_infer.add_argument('--dec_threshold', type=float, default=0.05, help='threshold when decoding output into boxes')
    parser_infer.add_argument('--nms_threshold', type=float, default=0.5, help='threshold when performing non-maximum suppression')
    parser_infer.add_argument('--topn', type=int, default=1000, help='top n boxes when decoding')
    parser_infer.add_argument('--ndetections', type=int, default=100, help='maximum boxes number after nms')
    parser_infer.add_argument('--checkpoint_dir', type=str, default='experiments/checkpoints', help='directory to save checkpoints in')
    parser_infer.add_argument('--result_dir', type=str, default='experiments/results', help='directory to save test results')

    return parser.parse_args()

def main(args):
    solver = Solver(args)
    if args.command == 'train':
        create_dirs(args.summary_dir, args.checkpoint_dir, args.log_dir)
        if args.dataset == 'coco':
            train_loader = CocoDataLoader(args.data_root, 'train2017', args.resize, args.max_size, 1, args.batch_size, is_training=True)
            val_loader = CocoDataLoader(args.data_root, 'val2017', args.resize, args.max_size, 1, args.batch_size, is_training=False)
        else:
            raise NotImplementedError('No such dataset')
        solver.train(train_loader, val_loader)

    elif args.command == 'infer':
        create_dirs(args.checkpoint_dir, args.result_dir)
        solver.inference(args.img_path)

if __name__ == '__main__':
    args = parse()
    main(args)