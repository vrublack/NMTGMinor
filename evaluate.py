import argparse
import os

import numpy as np
import torch

from nmtg.convert import load_checkpoint
from nmtg.custom_logging import add_log_options, setup_logging_from_args
from nmtg.options import add_general_options, add_task_option, add_trainer_option
from nmtg.tasks import Task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py")
    add_general_options(parser)
    task_class = add_task_option(parser)
    task_class.add_options(parser)
    trainer_class = add_trainer_option(parser)
    trainer_class.add_inference_options(parser)
    add_log_options(parser)

    parser.add_argument('-load_from', type=str, required=True,
                        help='Path to one or more pretrained models.')
    parser.add_argument('-output',
                        help="Path to output the predictions")

    args = parser.parse_args()

    logger = setup_logging_from_args(args, 'evaluate')

    logger.debug('Torch version: {}'.format(torch.__version__))
    logger.debug(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task = task_class.setup_task(args)  # type: Task

    logger.info('Loading checkpoint {}'.format(args.load_from))
    checkpoint = load_checkpoint(args.load_from)

    trainer = trainer_class(args, for_training=False, checkpoint=checkpoint)

    if args.discriminator:
        results, discr_accuracy = trainer.solve(task)
        logger.info(' | '.join(task.score_results(results) + ['Discriminator accuracy: {}'.format(discr_accuracy)]))

    else:
        results = trainer.solve(task)[0]
        logger.info(' | '.join(task.score_results(results)))

    if args.output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        task.save_results(results, args.output)
