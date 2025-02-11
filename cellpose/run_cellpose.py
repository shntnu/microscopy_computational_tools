import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
from cellpose import models
from cellpose.utils import masks_to_outlines
from cellpose.dynamics import get_centers
from scipy.ndimage import find_objects

SCALE = 2**16 - 1

def rescale(arr):
    arr = arr.copy()
    arr = arr.astype(np.double)
    arr -= np.min(arr)
    arr = (arr * SCALE / np.max(arr)).astype(np.uint16)
    return arr

def log_scale(arr):
    return rescale( np.log1p(arr) )
    return arr

def cell_outlines(images_folder, files):
    output = []
    model = models.Cellpose(gpu=True, model_type='nuclei')
    for file in files:
        try:
            im = Image.open(f'{images_folder}/{file}')
            im_resized = np.array(im.resize((512,512), Image.Resampling.NEAREST))
            if np.max(im_resized) - np.min(im_resized) < 1000:
                continue
            imgs = [log_scale(im_resized)]
            masks, flows, styles, diams = model.eval(imgs, diameter=25, channels=[0,0],
                                                     flow_threshold=0.4, do_3D=False)

            slices = find_objects(masks[0])
            # turn slices into array
            slices = np.array([
                np.array([i, si[0].start, si[0].stop, si[1].start, si[1].stop])
                for i, si in enumerate(slices)
                if si is not None
            ])
            centers, _ = get_centers(masks[0], slices)
            scale_x = im.size[0] / 512
            scale_y = im.size[1] / 512
            for center in centers:
                output.append( {'file': file, 'x': round(scale_x * center[1]), 'y': round(scale_y * center[0])} )
        except KeyboardInterrupt:
            sys.exit(0)
        except:
            print('failed', file)
    return output

if len(sys.argv) != 5:
  print('Usage: python run.py [path] [channel filter] [number of process] [process index]')
  exit()

images_folder   = sys.argv[1]
channel_filter  = sys.argv[2]
num_processes   = int(sys.argv[3])
process_idx     = int(sys.argv[4])
output_file     = f'cellpose_{process_idx}_{num_processes}.csv'

files = glob.glob(f'{images_folder}*{channel_filter}*')
files = [f.removeprefix(images_folder) for f in files]
files = sorted(files)[process_idx::num_processes]

output = cell_outlines(images_folder, files)

df = pd.DataFrame(output)
df = df.groupby('file').agg(list).reset_index()
df[['file', 'x', 'y']].to_csv(output_file, sep='\t', header=False, index=False)