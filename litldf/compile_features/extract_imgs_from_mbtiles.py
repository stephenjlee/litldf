import os, sys, argparse, json, hashlib
import logging
import geopandas as gpd
from landez import ImageExporter
from osgeo import gdal
from multiprocessing import Pool
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

# user defined imports
from litldf.utils import utils_gis as ug
from litldf.utils.utils_gis import create_circle_using_aep


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='')

    parser.add_argument('-ip', '--input_path',
                        default=os.path.join(os.environ.get('PROJECT_DATA_HDD'),
                                             'KEN',
                                             'KEN.geojson'),
                        help='the name of the dataset to run. See data_config.py for options')
    parser.add_argument('-mp', '--mbtiles_path',
                        default=r'/mnt/eighttb/code_8tb/arcgis-tile-downloader/out/KEN/KEN.mbtiles',
                        help='path to the mbtiles file')
    parser.add_argument('-z', '--zoom', type=int, default=17, help='')
    parser.add_argument('-jf', '--jpgs_folder',
                        default=os.path.join(os.environ.get('PROJECT_OUTEXT'), 'KEN-17'),
                        help='')
    parser.add_argument('-pps', '--par_pool_size', type=int, default=12, help='')

    return parser.parse_args(args)


# @cached_with_io
def load_cons_gdf(cons_path):
    cons_gdf = gpd.read_file(cons_path)
    return cons_gdf


def export_image(lon_west, lat_south, lon_east, lat_north, zoom_level, mbtiles_path, output_jpg_path):
    coords_hash = hashlib.sha256(json.dumps([lon_west, lat_south, lon_east, lat_north]).encode('utf-8')).hexdigest()

    # specify temporary png location
    output_img_png_path = os.path.join(os.environ.get("PROJECT_TEMP"), f'{coords_hash}.png')

    # this exports, but it doesn't actually adhere to the bounding box.
    # It only finds a tile that contains the bounding box.
    ie = ImageExporter(mbtiles_file=mbtiles_path, cache_scheme="wmts")
    # Bounding box format is (xmin, ymin, xmax, ymax)
    ie.export_image(bbox=(lon_west, lat_south, lon_east, lat_north),
                    zoomlevel=zoom_level,
                    imagepath=output_img_png_path)

    # need to clip now. Define the coordinates of this new image that was exported
    # get xyz of upper left corner
    upper_left_x, upper_left_y = ug.lat_lon_z_to_x_y(lat_north, lon_west, zoom_level)
    upper_left_lat, upper_lon_west = ug.x_y_z_to_lat_lon(upper_left_x, upper_left_y, zoom_level)
    # get xyz of bottom right corner
    bottom_right_x, bottom_right_y = ug.lat_lon_z_to_x_y(lat_south, lon_east, zoom_level)
    bottom_right_lat, bottom_lon_east = ug.x_y_z_to_lat_lon(bottom_right_x + 1, bottom_right_y + 1, zoom_level)

    # specify image bounds
    img_temp_bounds = gdal.Translate(
        '',
        output_img_png_path,
        outputBounds=[upper_lon_west, upper_left_lat, bottom_lon_east, bottom_right_lat],
        outputSRS='EPSG:4326',
        format='MEM'
    )

    # now clip back to desired coordinates.
    gdal.Translate(
        output_jpg_path,
        img_temp_bounds,
        projWin=[lon_west, lat_north, lon_east, lat_south],
        projWinSRS='EPSG:4326',
    )

    # remove temp file
    os.remove(output_img_png_path)

    print(f'ran gdal translate for {output_jpg_path}')


def get_image(tuple):
    (args, i, n_bldgs, bldg, jpgs_folder, zoom) = tuple

    output_jpg_path = os.path.join(jpgs_folder, f"{bldg['origin_origin_id']}.jpg")

    if Path(output_jpg_path).is_file():
        print(f'{output_jpg_path} already exists, skipping')
        return

    try:

        print(f'--------------------------------processing {i} of {n_bldgs}')

        lat = bldg.geometry.centroid.y
        lon = bldg.geometry.centroid.x

        logging.basicConfig(level=logging.DEBUG)

        # setting this to  give tiles of roughly 224 x 224 pixels.
        # We actually crop this image later (right before model
        # training) to make it correctly sized.

        window_side_length = 70 * (2 ** (17 - zoom))  # in meters

        circle_poly = create_circle_using_aep(lat, lon, 2 * window_side_length)
        lon_west = circle_poly.bounds[0]
        lat_south = circle_poly.bounds[1]
        lon_east = circle_poly.bounds[2]
        lat_north = circle_poly.bounds[3]

        export_image(lon_west,
                     lat_south,
                     lon_east,
                     lat_north,
                     args['zoom'],
                     args['mbtiles_path'],
                     output_jpg_path)


    except Exception as ex:

        output_txt_path = os.path.join(jpgs_folder, f"{bldg['origin_origin_id']}.txt")

        ex_str = f'{type(ex).__name__} - {ex.args[0]}'

        try:
            with open(output_txt_path, 'w') as f:
                f.write(ex_str)
        except:
            print(f'could not write to {output_txt_path}')

        print(f'exception: {ex_str}')

    return


def process_batch(args, output_gdf, batch_num):

    n_bldgs = output_gdf.shape[0]

    tuples = []
    for i, bldg in output_gdf.iterrows():
        tuples.append((args, i, n_bldgs, bldg, args['jpgs_folder'], args['zoom']))

    print(f'processed batch: {batch_num}')

    with Pool(args['par_pool_size']) as p:
        p.map(get_image, tuples)

def main(args):
    args = parse_args(args)
    args = args.__dict__

    os.makedirs(args['jpgs_folder'], exist_ok=True)
    os.makedirs(os.environ.get('PROJECT_TEMP'), exist_ok=True)

    batch_size = 100000
    batch_num = 0
    start = 0  # Initial start row

    while True:
        try:
            batch_num = batch_num + 1

            # Attempt to read a batch of the specified size
            batch = gpd.read_file(args['input_path'], rows=slice(start, start + batch_size))
            batch = batch[['origin_origin_id', 'geometry']]
            print('---------------------------------------')
            print(f"loaded gdf for batch: {batch_num}")
            print('---------------------------------------')

            if not batch.empty:
                process_batch(args, batch, batch_num)
                print('---------------------------------------')
                print(f"finished processing for batch: {batch_num}")
                print('---------------------------------------')

                start += batch_size

            else:
                # If the batch is empty, we've reached the end of the file
                break

        except StopIteration:
            # If a StopIteration is raised, we've reached the end of the file
            break

        except Exception as e:
            # Handle other potential exceptions (e.g., file not found, read errors)
            print(f"An error occurred: {e}")
            break


    print('finished!')



if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
