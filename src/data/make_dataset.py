# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from torchvision import datasets, transforms

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(output_filepath='~/MLOpsDTU/data/'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # exchange with the real mnist dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
    # Download data
    datasets.MNIST(output_filepath, download=True, train=True)
    datasets.MNIST(output_filepath, download=True, train=False)

    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    # Transform data
    datasets.MNIST(output_filepath, download=False, train=True, transform=transform)
    datasets.MNIST(output_filepath, download=False, train=False, transform=transform)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
