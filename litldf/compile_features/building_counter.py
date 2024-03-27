import os, sys
import numpy as np
import geopandas as gpd
from multiprocessing import Pool
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from litldf.utils.utils_gis import create_circle_using_aep
from litldf.compile_features.metadata_helper import MetadataLogger

class BuildingCounter(MetadataLogger):

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
    def generate_density_geoms(tuple):

        (i,
         unique_file_id,
         input_path,
         shapes_input_path,
         output_path,
         n_unique_file_ids,
         country_code) = tuple

        try:

            bldgs_in_buffer_geojson_path = os.path.join(input_path,
                                                        f"{unique_file_id}_geoms.geojson")
            grid_cell_cntry_intrsct_path = os.path.join(shapes_input_path, f"{unique_file_id}_shape.geojson")

            bldgs_in_buffer_gdf = gpd.read_file(bldgs_in_buffer_geojson_path)
            bldgs_in_buffer_sindex = bldgs_in_buffer_gdf.sindex

            grid_cell_cntry_intrsct = gpd.read_file(grid_cell_cntry_intrsct_path).iloc[0][0]

            # matches for smaller region
            sr_possible_matches_index = list(bldgs_in_buffer_sindex.intersection(grid_cell_cntry_intrsct.bounds))
            sr_possible_matches = bldgs_in_buffer_gdf.iloc[sr_possible_matches_index]
            sr_precise_matches = sr_possible_matches[sr_possible_matches.intersects(grid_cell_cntry_intrsct)]
            bldgs_in_smaller_region_gdf = sr_precise_matches.copy()

            n_bldgs_in_smaller_region = bldgs_in_smaller_region_gdf.shape[0]

            # for each building, find the number of other buildings within some distance
            cntry_num_bldgs_in_radius = []
            radius = 1000  # in m

            for j, bldg in enumerate(bldgs_in_smaller_region_gdf.iterrows()):
                if j % 100 == 0:
                    print(
                        f'analyzing bldg {j} of {n_bldgs_in_smaller_region} for cell {i} of {n_unique_file_ids} in {country_code}')

                circle_poly = create_circle_using_aep(bldg[1]['latitude'], bldg[1]['longitude'], radius)

                possible_matches_index = list(bldgs_in_buffer_sindex.intersection(circle_poly.bounds))
                possible_matches = bldgs_in_buffer_gdf.iloc[possible_matches_index]

                # without cuda
                num_bldgs_in_radius = np.sum(possible_matches.intersects(circle_poly))

                cntry_num_bldgs_in_radius.append(num_bldgs_in_radius)

            bldgs_in_smaller_region_gdf['n_bldgs_1km_away'] = cntry_num_bldgs_in_radius

            try:
                bldgs_in_smaller_region_gdf.to_file(
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


            input_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code, self.args['previous_class_name'])
            shapes_input_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code, self.args['shapes_class_name'])

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
                    print(f'added {unique_file_id} to the list for counting geometries.')
                    par_list.append((i,
                                     unique_file_id,
                                     input_path,
                                     shapes_input_path,
                                     output_path,
                                     n_unique_file_ids,
                                     country_code))

                    self.outputs[completed_path] = 'x'

            with Pool(self.args['par_pool_size']) as p:
                print(p.map(self.generate_density_geoms, par_list, chunksize=1))



    def count_bldgs_parallel(self):
        if self.outputs_exist():
            print("BuildingCounter: all expected counted building files exist. Skipping counting.")
            return
        self.process()
        self.save_metadata()

