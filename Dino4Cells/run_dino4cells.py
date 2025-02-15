import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
from simple_dino_embed import dino_model
from ast import literal_eval
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

SCALE = 2**16 - 1

def rescale(arr, axes=[-1, -2]):
    arr = arr.copy()
    arr = arr.astype(np.float32)
    arr -= np.min(arr)
    arr = (arr * SCALE / (np.max(arr) + 1e-8))
    return arr

class Data_Set(Dataset): 
    def __init__(self, images_folder, channel_filters, centers, subwindow=128):
        self.channel_filters = channel_filters
        self.subwindow = subwindow
        self.images_folder = images_folder
        self.file_loaded = None
        self.images = None
        self.centers = centers
        cells_per_image = centers.x.str.len()
        self.num_cells = sum(cells_per_image)
        self.cell_to_image = [idx for idx, cells_in_image in enumerate(cells_per_image) for _ in range(cells_in_image)]
        self.cell_to_offset = [j for cells_in_image in cells_per_image for j in range(cells_in_image)]
    def __len__(self): 
        return self.num_cells
    def extract_subimage(self, images, center_x, center_y, subwindow=128):
        output = np.zeros((len(self.channel_filters), subwindow, subwindow), dtype=np.float32)  # Batch, channels, width, height
        for ichan, im in enumerate(self.images):
            output[ichan, ...] = im[center_x:center_x + subwindow, center_y:center_y + subwindow]
        # rescale each subimage
        output -= output.min(axis=(1, 2), keepdims=True)
        out_max = output.max(axis=(1, 2), keepdims=True)
        out_max[out_max == 0] = 1
        output /= out_max
        return output
    def __getitem__(self, idx):
        image_idx = self.cell_to_image[idx]
        if self.file_loaded != image_idx:
            self.file_loaded = image_idx
            self.images = []
            DNA_filter = self.channel_filters[0]
            for channel_filter in self.channel_filters:
                filename = self.images_folder + self.centers.index[image_idx].replace(DNA_filter, channel_filter)
                self.images.append( np.asarray(Image.open(filename)) )
            # pad images to simplify subimage extraction
            padwidth = self.subwindow // 2
            # this makes x, y the upper left corner of each subwindow, and prevents out-of-bounds
            self.images = [np.pad(im, (padwidth, self.subwindow - padwidth)) for im in self.images]
        center_x = self.centers['x'].iloc[image_idx][ self.cell_to_offset[idx] ]
        center_y = self.centers['y'].iloc[image_idx][ self.cell_to_offset[idx] ]
        cells = self.extract_subimage(self.images, center_x, center_y, subwindow=128)
        return self.centers.index[image_idx], center_x, center_y, cells


class Batch_Sampler(Sampler):
    def __init__(self, centers):
        self.cells_per_image = centers.x.str.len()
    def __iter__(self):
        offset = 0
        for batch_size in self.cells_per_image:
            yield [offset+i for i in range(batch_size)]
            offset += batch_size



def extract_subimages(images, centers_x, centers_y, subwindow=128):
    # pad images to simplify subimage extraction
    padwidth = subwindow // 2
    # this makes x, y the upper left corner of each subwindow, and prevents out-of-bounds
    images = [np.pad(im, (padwidth, subwindow - padwidth)) for im in images]

    output = np.zeros((len(centers_x), 5, subwindow, subwindow), dtype=np.float32)  # Batch, channels, width, height
    for ix, (x, y) in enumerate(zip(centers_x, centers_y)):
        for ichan, im in enumerate(images):
            output[ix, ichan, ...] = im[y:y + subwindow, x:x + subwindow]

    # rescale each subimage
    output -= output.min(axis=(2, 3), keepdims=True)
    out_max = output.max(axis=(2, 3), keepdims=True)
    out_max[out_max == 0] = 1
    output /= out_max
    return output


def cell_embeddings(images_folder, centers, channel_filters, num_workers):
    output = []
    model = dino_model(args.model_path)

    ds = Data_Set(images_folder, channel_filters, centers)
    bs = Batch_Sampler(centers)
    dataloader = DataLoader(ds, batch_sampler=bs, num_workers=num_workers, pin_memory=True)
    for filename, centers_x, centers_y, subimages in dataloader:
        centers_x = centers_x.tolist()
        centers_y = centers_y.tolist()
        embeddings = model(subimages)
        for x, y, em in zip(centers_x, centers_y, embeddings):
            output.append([filename[0], x, y, em])
    return output


parser = argparse.ArgumentParser(description='run_dino4cells', prefix_chars='@')
parser.add_argument('model_path', type=str, help='model')
parser.add_argument('plate_path', type=str, help='folder containing images')
parser.add_argument('channel_filters', type=str, nargs=5, help='substring in filename to identify DNA, RNA, AGP, ER and Mito image, respectively')
parser.add_argument('centers_path', type=str, help='filename with cell centers')
parser.add_argument('num_workers', type=int, help='number of processes for loading data')
args = parser.parse_args()

output_file     = f'dino4cells.tsv'

images_folder = args.plate_path
if not images_folder.endswith('/'):
    images_folder = images_folder + '/'

centers = pd.read_table(args.centers_path, converters={'x':literal_eval, 'y':literal_eval}, index_col='file')

output = cell_embeddings(images_folder, centers, args.channel_filters, args.num_workers)

df = pd.DataFrame(output, columns="file x y embedding".split())
df.to_csv(output_file, sep="\t", index=None)