import sys
import glob
import os.path
import numpy as np
import pandas as pd
from dino4cells.simple_embed import dino_model
from ast import literal_eval
import argparse
from torch.utils.data import DataLoader
from image_loader import Data_Set, Batch_Sampler

def cell_embeddings(images_folder, centers, channel_filters, num_workers):
    output = []
    model = dino_model(args.model_path)

    # generate image names
    dna_images = [images_folder + imname for imname in centers.index]
    other_images = [[dnaim.replace(channel_filters[0], newchan) for dnaim in dna_images]
                    for newchan in channel_filters[1:]]
    image_groups = list(zip(dna_images, *other_images))

    # check that image paths point to actual files
    actual_files = set(glob.glob(images_folder + '*'))
    image_groups = [imgrp for imgrp in image_groups if all(filepath in actual_files for filepath in imgrp)]
    if len(image_groups) != len(centers):
        print(f"WARNING: Found complete image sets for {len(image_groups)} out of {len(centers)}")

    ds = Data_Set(image_groups, centers)
    bs = Batch_Sampler(image_groups, centers)
    dataloader = DataLoader(ds, batch_sampler=bs, num_workers=num_workers, pin_memory=True)

    # generate embeddings
    for dna_imnames, centers_x, centers_y, subimages in dataloader:
        dna_imname = dna_imnames[0]
        centers_x = centers_x.tolist()
        centers_y = centers_y.tolist()
        print(dna_imname, len(centers_x))
        embeddings = model(subimages)
        for x, y, em in zip(centers_x, centers_y, embeddings):
            output.append([dna_imname, x, y, em])
    return output


parser = argparse.ArgumentParser(description='run_dino4cells', prefix_chars='@')
parser.add_argument('model_path', type=str, help='model')
parser.add_argument('plate_path', type=str, help='folder containing images')
parser.add_argument('channel_filters', type=str, nargs=5, help='substring in filename to identify DNA, RNA, AGP, ER and Mito image, respectively')
parser.add_argument('centers_path', type=str, help='filename with cell centers')
parser.add_argument('num_workers', type=int, help='number of processes for loading data')
parser.add_argument('output_file', type=str, help='output filename', nargs='?', default='dino4cells.tsv')
args = parser.parse_args()

images_folder = args.plate_path
if not images_folder.endswith('/'):
    images_folder = images_folder + '/'

centers = pd.read_table(args.centers_path, converters={'x':literal_eval, 'y':literal_eval}, index_col='file')

output = cell_embeddings(images_folder, centers, args.channel_filters, args.num_workers)

df = pd.DataFrame(output, columns="file x y embedding".split())
df.to_csv(args.output_file, sep="\t", index=None)