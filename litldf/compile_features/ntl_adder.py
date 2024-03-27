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
    add_class_value_and_local_histogram_to_gdf, \
    add_feat_value_and_local_means_to_gdf

from litldf.compile_features.metadata_helper import MetadataLogger


class NtlAdder(MetadataLogger):
    ntl_raster = None

    def __init__(self,
                 args,
                 **kwargs
                 ):

        args.update(kwargs)

        self.args = args
        super().__init__(os.path.join(os.environ.get("PROJECT_CACHE_METADATA"),
                                      self.__class__.__name__),
                         self.args)

        self.window_sizes = json.loads(self.args['window_sizes'])

        NtlAdder.ntl_raster = rio.open(self.args['ntl_path'])

    @staticmethod
    def add_ntl(tuple):

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

            bldgs_gdf = add_feat_value_and_local_means_to_gdf(bldgs_gdf,
                                                              NtlAdder.ntl_raster,
                                                              props.args['ntl_year'],
                                                              props.args['feat_name'],
                                                              window_sizes=props.window_sizes)

            try:
                print(f'wrote files for grid cell: {unique_file_id}')
                bldgs_gdf.to_file(
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
                    print(f'added {unique_file_id} to the list for generating ntl geometries.')
                    par_list.append((i,
                                     unique_file_id,
                                     input_path,
                                     output_path,
                                     n_unique_file_ids,
                                     country_code,
                                     self))

                    self.outputs[completed_path] = 'x'

            with Pool(self.args['par_pool_size']) as p:
                print(p.map(self.add_ntl, par_list, chunksize=1))


    def add_ntl_parallel(self):
        if self.outputs_exist():
            print("NtlAdder: all expected ntl added building files exist. Skipping adding.")
            return
        self.process()
        self.save_metadata()
