import hailtop.batch as hb
from shlex import quote
import yaml
import argparse
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description='run dino4cells on Hail Batch', prefix_chars='@')
parser.add_argument('plate_path', type=str, help='folder containing plates')
parser.add_argument('output_folder', type=str, help='output folder')
parser.add_argument('channel_filters', type=str, nargs=5, help='substring in filename to identify DNA, RNA, AGP, ER and Mito image, respectively')
parser.add_argument('centers_path', type=str, help='path to cell centers')
parser.add_argument('plates', type=str, nargs='+', help='plate names')
args = parser.parse_args()

plate_path = urlparse(args.plate_path)
bucket_name = plate_path.netloc
input_folder = plate_path.path.rstrip('/')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

backend = hb.ServiceBackend(billing_project=config['hail-batch']['billing-project'],
                            remote_tmpdir=config['hail-batch']['remote-tmpdir'],
                            regions=config['hail-batch']['regions'])

b = hb.Batch(backend=backend, name='dino4cell')
for plate in args.plates:
    j = b.new_job(name=f'dino4cell {plate}')
    j.cloudfuse(bucket_name, '/images')
    j._machine_type = config['dino4cells']['machine-type']
    j.image(config['dino4cells']['docker-image'])
    j.storage('20Gi') # should be large enough for Docker image, model and tsv output (not for images)


    dino4cells_model = b.read_input(config['dino4cells']['model'])
    dino4cells_code = b.read_input(config['dino4cells']['code'])
    centers_file = b.read_input(args.centers_path.format(plate=plate))

    num_workers = config['dino4cells']['num-workers']
    image_folder = f'{input_folder}/{plate}/'

    j.command(f'tar -zxf {dino4cells_code}')
    j.command(f'python3 run_dino4cells.py {dino4cells_model} /images/{quote(image_folder)} {[quote(f) for f in args.channel_filters]} {quote(centers_file)} {num_workers}')
    j.command(f'mv dino4cells.tsv {j.ofile}')
    b.write_output(j.ofile, f'{args.output_folder}/dino4cells_{plate}.tsv')
b.run() 
