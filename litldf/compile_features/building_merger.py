import os, sys, hashlib
import numpy as np
import geopandas as gpd
from multiprocessing import Pool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from litldf.utils import utils_vector
from litldf.compile_features.metadata_helper import MetadataLogger


class BuildingMerger(MetadataLogger):

    def __init__(self,
                 args,
                 **kwargs
                 ):

        args.update(kwargs)
        self.args = args
        super().__init__(os.path.join(os.environ.get("PROJECT_CACHE_METADATA"),
                                      self.__class__.__name__),
                         self.args)

    @staticmethod
    def merge_bldgs(tuple):

        (i,
         unique_file_id,
         google_input_path,
         microsoft_input_path,
         output_path,
         n_unique_file_ids,
         country_code) = tuple

        try:

            google_bldgs_in_buffer_geojson_path = os.path.join(google_input_path,
                                                        f"{unique_file_id}_geoms.geojson")
            microsoft_bldgs_in_buffer_geojson_path = os.path.join(microsoft_input_path,
                                                        f"{unique_file_id}_geoms.geojson")

            try:
                bldgs_in_buffer_gdf = gpd.read_file(google_bldgs_in_buffer_geojson_path)
            except:
                bldgs_in_buffer_gdf = None

            try:
                microsoft_bldgs_in_buffer_gdf = gpd.read_file(microsoft_bldgs_in_buffer_geojson_path)
                n_msft_bldgs = microsoft_bldgs_in_buffer_gdf.shape[0]

                bldgs_in_buffer_sindex = bldgs_in_buffer_gdf.sindex
                bldgs_in_buffer_ids_to_remove = []

                # this is an invovled step. Loop through all Microsoft buildings, see if they overlap
                # buildings that currently exist in the google open buildings data, and if so:
                # remove all corresponding buildings and add a new entry.
                microsoft_bldg_dict_array = []
                for j, msft_bldg in enumerate(microsoft_bldgs_in_buffer_gdf.iterrows()):

                    if j % 10 == 0:
                        print(f'querying msft_bldg {j} of {n_msft_bldgs} using pid {os.getpid()}')

                    msft_query_geom = msft_bldg[1]['geometry']

                    possible_matches_index = list(bldgs_in_buffer_sindex.intersection(msft_query_geom.bounds))
                    possible_matches = bldgs_in_buffer_gdf.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(msft_query_geom)]

                    percent_google_overlap = 0.0
                    if precise_matches.shape[0] > 0:
                        precise_matches_dissolved = precise_matches.dissolve()
                        precise_matches_clipped = gpd.clip(precise_matches_dissolved, msft_query_geom)
                        percent_google_overlap = utils_vector.geom_to_area(
                            precise_matches_clipped.iloc[0]['geometry']) / utils_vector.geom_to_area(msft_query_geom)

                    if percent_google_overlap <= 0.01:
                        msft_bldg_dict = {
                            'latitude': msft_bldg[1]['geometry'].centroid.xy[0][0],
                            'longitude': msft_bldg[1]['geometry'].centroid.xy[1][0],
                            'area_in_meters': utils_vector.geom_to_area(msft_bldg[1]['geometry']),
                            'confidence': -1.0,
                            'origin_id': hashlib.sha256(msft_bldg[1]['geometry'].wkt.encode('utf-8')).hexdigest()[0:20],
                            'origin': 'msft',
                            'geometry': msft_bldg[1]['geometry'],
                        }
                        microsoft_bldg_dict_array.append(msft_bldg_dict)
            except:
                print('Cannot read Microsoft buildings in buffer! It might be empty! Not adding it to the final file.')
                microsoft_bldgs_in_buffer_gdf = None

            if (bldgs_in_buffer_gdf is not None) and (microsoft_bldgs_in_buffer_gdf is not None):
                bldgs_in_buffer_gdf = bldgs_in_buffer_gdf.append(microsoft_bldg_dict_array, ignore_index=True)
            elif (bldgs_in_buffer_gdf is None) and (microsoft_bldgs_in_buffer_gdf is not None):
                bldgs_in_buffer_gdf = microsoft_bldgs_in_buffer_gdf

            try:
                bldgs_in_buffer_gdf.to_file(
                    os.path.join(output_path, f"{unique_file_id}_geoms.geojson"),
                    driver='GeoJSON')
            except:
                print('Cannot write dataframe to file! It might be empty!')
                with open(os.path.join(output_path, f"{unique_file_id}_geoms.err"), 'w') as f:
                    f.write('Cannot write dataframe to file! It might be empty!')

            with open(os.path.join(output_path, f"{unique_file_id}_completed.txt"), 'w') as f:
                f.write('Completed!')

        except:
            print(f'Something went wrong with {unique_file_id}. Debug for more info! \n')

    def process(self):

        par_list = []
        for country_code in self.args['country_codes']:

            google_input_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code,
                                             self.args['google_class_name'])
            microsoft_input_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code,
                                                self.args['microsoft_class_name'])
            output_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code, self.__class__.__name__)
            os.makedirs(output_path, exist_ok=True)

            all_files = np.array(os.listdir(microsoft_input_path))

            if all_files.size == 0:
                print(f'No files in {microsoft_input_path}. Skipping.')
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
                    print(f'added {unique_file_id} to the list for merging geometries.')
                    par_list.append((i,
                                     unique_file_id,
                                     google_input_path,
                                     microsoft_input_path,
                                     output_path,
                                     n_unique_file_ids,
                                     country_code))

                    self.outputs[completed_path] = 'x'

            with Pool(self.args['par_pool_size']) as p:
                print(p.map(self.merge_bldgs, par_list, chunksize=1))

    def merge_bldgs_parallel(self):
        if self.outputs_exist():
            print("BuildingCounter: all expected counted building files exist. Skipping counting.")
            return
        self.process()
        self.save_metadata()
