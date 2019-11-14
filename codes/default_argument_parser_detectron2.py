
import argparse
import os

def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description='Learning Detectron2:train_net.py')
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='whether to attempt to resume from the checkpoint directory'
    )
    parser.add_argument('--eval-only', action='store_true', help='perform evaluation only')
    parser.add_argument('--num-gpus', type=int, default=1, help='number of gpus *per machine*')
    parser.add_argument('--num-machines', type=int, default=1)
    parser.add_argument(
        '--machine-rank', type=int, default=0, help='the rank of this machine (unique per machine)'
        )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument('--dist-url', default=f'tcp://127.0.0.1:{port}')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == '__main__':
    args = default_argument_parser().parse_args()