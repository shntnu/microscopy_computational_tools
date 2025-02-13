import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
from simple_dino_embed import dino_model
from ast import literal_eval

SCALE = 2**16 - 1

def rescale(arr, axes=[-1, -2]):
    arr = arr.copy()
    arr = arr.astype(np.float32)
    arr -= np.min(arr)
    arr = (arr * SCALE / (np.max(arr) + 1e-8))
    return arr

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


def cell_embeddings(images_folder, file_groups, centers):
    output = []
    model = dino_model()

    for filegrp in file_groups:
        print(filegrp[0])
        try:
            images = [Image.open(f'{images_folder}/{file}') for file in filegrp]
            centers_info =  centers.loc[filegrp[0]]
            subimages = extract_subimages(images, centers_info.x, centers_info.y)
            embeddings = model(subimages)
            for x, y, em in zip(centers_info.x, centers_info.y, embeddings):
                output.append([filegrp[0], x, y, em])
        except KeyboardInterrupt:
            sys.exit(0)
#         except Exception as e:
#             import pdb
#             pdb.set_trace()
#             print('failed', filegrp, str(e))
    return output

if len(sys.argv) != 10:
    # expected channel order is DNA, RNA, AGP, ER, Mito
    print('Usage: python run.py [plate path] [DNA substring] [RNA substring] [AGP substring] [ER substring] [Mito substring] [centers_path] [number of process] [process index]')
    exit()

images_folder   = sys.argv[1]
channel_filters = sys.argv[2:7]
centers_file = sys.argv[7]
num_processes   = int(sys.argv[8])
process_idx     = int(sys.argv[9])
output_file     = f'dino4cells_{process_idx}_{num_processes}.tsv'

if not images_folder.endswith('/'):
    images_folder = images_folder + '/'

DNA_filter = channel_filters[0]
DNA_files = glob.glob(f'{images_folder}*{DNA_filter}*')
centers = pd.read_table(centers_file, converters={'x':literal_eval, 'y':literal_eval}, index_col='file')

DNA_files = [f.removeprefix(images_folder) for f in DNA_files]
file_groups = [[f.replace(DNA_filter, chanfilt) for chanfilt in channel_filters] for f in DNA_files]
file_groups = sorted(file_groups)[process_idx::num_processes]

output = cell_embeddings(images_folder, file_groups, centers)

df = pd.DataFrame(output, columns="file x y embedding".split())
df.to_csv(output_file, sep="\t", index=None)

