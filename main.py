import argparse

import torch

from src.predict import has_tomatoes
from src.train import train as train_


def train(imgs_root, ann_path, map_path, out_cp_path, device):
    model = train_(imgs_root, ann_path, map_path, device)
    torch.save(model.state_dict(), out_cp_path)


def predict(img_path, cp_path):
    res = has_tomatoes(img_path, cp_path)
    print(res)


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument(
        '--imgs_root',
        type=str,
        help='Path of the image folder',
        required=True
    )
    parser_train.add_argument(
        '--ann_path',
        type=str,
        help='Path of the annotation file',
        required=True
    )
    parser_train.add_argument(
        '--map_path',
        type=str,
        help='Path of the mapping file',
        required=True
    )
    parser_train.add_argument(
        '--out_cp_path',
        type=str,
        help='The model checkpoint out path',
        required=True
    )
    parser_train.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use',
        required=False,
        default='cuda'
    )

    parser_predict = subparsers.add_parser(
        'predict',
        help='Predict for an image'
    )
    parser_predict.add_argument(
        '--img_path',
        type=str,
        help='Path of the image',
        required=True
    )
    parser_predict.add_argument(
        '--cp_path',
        type=str,
        help='Path of the model checkpoint',
        required=True
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.command == 'predict':
        predict(args.img_path, args.cp_path)
    elif args.command == 'train':
        train(
            args.imgs_root,
            args.ann_path,
            args.map_path,
            args.out_cp_path,
            args.device
        )
