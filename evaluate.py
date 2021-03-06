import sys
import argparse

from utils import load_texts

sys.path.append('video_description_eval/coco-caption')
from video_description_eval.evaluate import score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics between generation and references')
    parser.add_argument('-gen', '--generation', type=str, default='results/predictions.txt',
                        help='Set the path to gnerated texts list (default is results/predictions.txt).')
    parser.add_argument('-ref', '--references', type=str, default='results/references.txt',
                        help='Set the path to reference texts list (default is data/references.txt).')

    args = parser.parse_args()

    metrics_results = score(load_texts(args.references), load_texts(args.generation))

    log_msg = "RESULTS: "
    for name, result in metrics_results.items():
    	log_msg += '{0}: {1:.3f} '.format(name, result)
    print(log_msg)