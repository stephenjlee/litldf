import sys, os, json
import numpy as np
from multiprocessing import Pool
import geopandas as gpd
import rasterio as rio

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from litldf.utils.utils_raster import \
    add_utm_cols_to_latlon_gdf, \
    add_class_value_and_local_histogram_to_gdf
from litldf.compile_features.metadata_helper import MetadataLogger


class LulcAdder(MetadataLogger):

    def __init__(self,
                 args,
                 **kwargs
                 ):

        args.update(kwargs)

        self.args = args
        super().__init__(os.path.join(os.environ.get("PROJECT_CACHE_METADATA"),
                                      self.__class__.__name__),
                         self.args)

        self.args['lulc_classes'] = json.loads(self.args['lulc_classes'])
        self.args['lulc_feature_shares_to_add'] = json.loads(self.args['lulc_feature_shares_to_add'])
        self.args['window_sizes'] = json.loads(self.args['window_sizes'])

        self.lulc_utm_tifs = np.array(os.listdir(self.args['lulc_utm_path']))
        self.lulc_utm_tifs_first_three = np.array([name[0:name.find('_')] for name in self.lulc_utm_tifs.tolist()])


    @staticmethod
    def add_lulc(tuple):

        (i,
         unique_file_id,
         input_path,
         output_path,
         n_unique_file_ids,
         country_code,
         props) = tuple

        try:

            input_geoms_geojson_path = os.path.join(input_path,
                                                      f"{unique_file_id}_geoms.geojson")

            bldgs_gdf = gpd.read_file(input_geoms_geojson_path)
            bldgs_gdf = add_utm_cols_to_latlon_gdf(bldgs_gdf)

            all_utm_zones = np.unique(bldgs_gdf['zone_number_letters'].values)

            for i, utm_zone in enumerate(all_utm_zones):
                print(f'processing for UTM zone {utm_zone}')

                tif_name = props.lulc_utm_tifs[props.lulc_utm_tifs_first_three == utm_zone][0]

                # load specific UTM raster
                lulc_raster = rio.open(os.path.join(props.args['lulc_utm_path'], tif_name))

                # ensuring the polygons match the raster UTM zone
                bldgs_gdf_filtered = bldgs_gdf[(bldgs_gdf['zone_number_letters'] == utm_zone)].copy()
                bldgs_gdf_filtered = add_class_value_and_local_histogram_to_gdf(bldgs_gdf_filtered,
                                                                                lulc_raster,
                                                                                props.args['lulc_year'],
                                                                                props.args['lulc_classes'],
                                                                                props.args['lulc_feature_shares_to_add'],
                                                                                'lulc',
                                                                                window_sizes=props.args['window_sizes'])

                if i == 0:
                    combined_gdf = bldgs_gdf_filtered
                else:
                    combined_gdf = combined_gdf.append(bldgs_gdf_filtered, ignore_index=True)

            try:
                print(f'wrote files for grid cell: {unique_file_id}')
                combined_gdf.to_file(
                    os.path.join(output_path, f"{unique_file_id}_geoms.geojson"),
                    driver='GeoJSON')
            except:
                print(f'Cannot write dataframe to file! It might be empty! Failed on grid cell: {unique_file_id}')
                with open(os.path.join(output_path, f"{unique_file_id}_geoms.err"), 'w') as f:
                    f.write('Cannot write dataframe to file! It might be empty!')

            with open(os.path.join(output_path, f"{unique_file_id}_completed.txt"), 'w') as f:
                f.write('Completed!')

        except:
            print(f'Something went wrong with {unique_file_id}. Debug for more info! \n')


    def process(self):

        par_list = []
        for country_code in self.args['country_codes']:

            input_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code,
                                      self.args['previous_class_name'])
            output_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code, self.__class__.__name__)
            os.makedirs(output_path, exist_ok=True)

            all_files = np.array(os.listdir(input_path))

            if all_files.size == 0:
                print(f'No files in {input_path}. Skipping.')
                continue

            all_files_underscore_index = np.char.find(all_files, '_completed.txt')
            all_files = all_files[all_files_underscore_index > 0]
            all_files_underscore_index = np.char.find(all_files, '_completed.txt')
            all_files_ids = np.char.ljust(all_files, all_files_underscore_index)
            unique_file_ids = np.unique(all_files_ids)
            n_unique_file_ids = unique_file_ids.size

            for i, unique_file_id in enumerate(unique_file_ids):

                completed_path = os.path.join(output_path, f"{unique_file_id}_completed.txt")

                # if the files already exist, skip it. i.e. only run if there's a missing file
                if not os.path.exists(completed_path):
                    print(f'added {unique_file_id} to the list for generating lulc geometries.')
                    par_list.append((i,
                                     unique_file_id,
                                     input_path,
                                     output_path,
                                     n_unique_file_ids,
                                     country_code,
                                     self))

                    self.outputs[completed_path] = 'x'

            with Pool(self.args['par_pool_size']) as p:
                print(p.map(self.add_lulc, par_list, chunksize=1))

    def add_lulc_parallel(self):
        if self.outputs_exist():
            print("LulcAdder: all expected lulc added building files exist. Skipping adding.")
            return
        self.process()
        self.save_metadata()
