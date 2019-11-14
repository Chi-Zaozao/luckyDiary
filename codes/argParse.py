# %load ./codes/argParse.py
import argparse
parser = argparse.ArgumentParser()
group =  parser.add_mutually_exclusive_group()
# calling this program now need to specify 2 argument 'echo' and 'square'
# Positional arguments
parser.add_argument('square', type = int, help = 'display a square of a given number')
# Optional arguments
group.add_argument('-n','--noise', action='store_true',
                   help='make some noise!')
group.add_argument('-q', '--quiet', action='store_true',
                   help = 'be qiuet!')
parser.add_argument('-v','--verbosity', help = 'increase output verbosity', choices = [1, 2, 3], type = int)
parser.add_argument('-c','--count', default = 100, 
                    help = 'count  the number of occurrences of a specific optional arguments', action = 'count')
args = parser.parse_args()
answer = args.square**2
if args.count:
    print(f'It counts {args.count}')
    print(type(args.count))
if args.verbosity:
    print(f'verbosity {args.verbosity} turned on')
if args.noise:
    print('the square of {} equals {}'.format(args.square, answer))
elif args.quiet:
    print(answer)
else:
    print(f'{answer} !!!')