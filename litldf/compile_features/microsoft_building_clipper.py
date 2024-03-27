import os, sys, random, time, zipfile, urllib.request
from multiprocessing import Pool
import geopandas as gpd
import pandas as pd
import numpy as np

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from litldf.utils import utils_vector
from litldf.compile_features.metadata_helper import MetadataLogger


class MicrosoftBuildingClipper(MetadataLogger):

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
    def extract_microsoft(tuple):
        """
        Return a list of random ingredients as strings.

        :param kind: Optional "kind" of ingredients.
        :type kind: list[str] or None
        :raise lumache.InvalidKindError: If the kind is invalid.
        :return: The ingredients list.
        :rtype: list[str]

        """
        import copy
        import numpy as np
        import geopandas as gpd

        (grid_cell_id,
         input_path,
         output_path,
         cntry_poly_wkt,
         microsoft_cells_path,
         microsoft_buildings_path) = tuple

        # create logger

        print(f'starting to merge osm and google for grid_cell_id: {grid_cell_id}')

        # setup

        grid_cell_geometry = gpd.read_file(os.path.join(input_path, f"{grid_cell_id}_shape.geojson"))
        grid_cell_geometry = grid_cell_geometry.iloc[0].geometry

        microsoft_cells_df = gpd.read_file(microsoft_cells_path)
        # define output files
        bldgs_in_buffer_geojson_path = os.path.join(output_path,
                                                    f"{grid_cell_id}_geoms.geojson")


        buffer_radius = 1200  # in m
        buffered_grid_cell_cntry_intrsct = utils_vector.buffer_polygon_using_aep(grid_cell_geometry,
                                                                                 buffer_radius)

        # now see what microsoft cells overlap the buffered grid cell
        intersecting_microsoft_cells_df = microsoft_cells_df[
            microsoft_cells_df.intersects(buffered_grid_cell_cntry_intrsct)]
        n_intersecting_microsoft_cells = intersecting_microsoft_cells_df.shape[0]

        try:
            print('starting to load goog bldgs')

            if intersecting_microsoft_cells_df.shape[0] == 0:
                print('No intersecting google tiles in cell')
                bldgs_in_buffer_gdf = pd.DataFrame()


            matching_quadkey_paths = []

            # construct dataframe for all buildings that may intersect that grid cell
            for i, microsoft_cell in enumerate(intersecting_microsoft_cells_df.iterrows()):
                print(
                    f'---loading google tile {i + 1} of {n_intersecting_microsoft_cells} for grid cell {grid_cell_id}')

                quadkey = str(microsoft_cell[1]['quadkey'])

                while len(quadkey) > 0:

                    microsoft_buildings_geojson = os.path.join(microsoft_buildings_path,
                                                               f"{quadkey}.geojson")

                    if os.path.exists(microsoft_buildings_geojson):
                        matching_quadkey_paths.append(microsoft_buildings_geojson)
                        break

                    else:
                        quadkey = quadkey[:-1]

            matching_quadkey_paths = np.unique(matching_quadkey_paths)

            for i, quadkey_path in enumerate(matching_quadkey_paths):
                print(
                    f'---loading google buildings {i + 1} of {matching_quadkey_paths.shape[0]} for grid cell {grid_cell_id}')

                bldgs_gdf = gpd.read_file(quadkey_path)

                if i == 0:
                    all_bldgs_gdf = copy.deepcopy(bldgs_gdf)
                else:
                    all_bldgs_gdf = all_bldgs_gdf.append(bldgs_gdf)

            bldgs_in_buffer_gdf = all_bldgs_gdf[all_bldgs_gdf.intersects(buffered_grid_cell_cntry_intrsct)]

            del all_bldgs_gdf
            # before starting anything, add a new column to google open buildings data
            n_goog_bldgs_in_buffer = bldgs_in_buffer_gdf.shape[0]

            bldgs_in_buffer_gdf = bldgs_in_buffer_gdf.assign(
                origin=np.repeat('Microsoft', n_goog_bldgs_in_buffer)
            )
            print('done with google bldgs for now')
        except:
            print('Failed to load google bldgs in cell')

        print('trying to write buildings buffer to file')

        try:
            # write buildings within the buffer to csv:
            bldgs_in_buffer_gdf.to_file(bldgs_in_buffer_geojson_path, driver='GeoJSON')

            del bldgs_in_buffer_gdf
        except:
            print('Cannot write dataframe to file! It might be empty!')
            with open(os.path.join(output_path, f"{grid_cell_id}_geoms.err"), 'w') as f:
                f.write('Cannot write dataframe to file! It might be empty!')

        with open(os.path.join(output_path, f"{grid_cell_id}_completed.txt"), 'w') as f:
            f.write('Completed!')

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

                # load all countries shapes
                countries_shapes_df = gpd.read_file(self.args['countries_shapes_path'])

                # identify country shapely polygon of interest
                cntry_poly = countries_shapes_df[countries_shapes_df['ISO3'] == country_code]['geometry'].iloc[0]
                cntry_poly_wkt = cntry_poly.wkt

                # if the files already exist, skip it. i.e. only run if there's a missing file
                if not os.path.exists(completed_path):
                    print(f'added {unique_file_id} to the list for generating density geometries.')
                    par_list.append((unique_file_id,
                                     input_path,
                                     output_path,
                                     cntry_poly_wkt,
                                     self.args['microsoft_cells_path'],
                                     self.args['microsoft_buildings_path']))

                    self.outputs[completed_path] = 'x'

            with Pool(self.args['par_pool_size']) as p:
                print(p.map(self.extract_microsoft, par_list, chunksize=1))

    def clip_bldgs_parallel(self):
        if self.outputs_exist():
            print("BuildingClipper: all expected clipped building geojsons exist. Skipping clipping.")
            return
        self.process()
        self.save_metadata()
