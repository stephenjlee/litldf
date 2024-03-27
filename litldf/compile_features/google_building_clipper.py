import os, sys, random, time, zipfile, urllib.request
from multiprocessing import Pool
import geopandas as gpd
import pandas as pd

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from litldf.utils import utils_vector
from litldf.compile_features.metadata_helper import MetadataLogger


class GoogleBuildingClipper(MetadataLogger):

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
    def extract_google(tuple):
        """
        Return a list of random ingredients as strings.

        :param kind: Optional "kind" of ingredients.
        :type kind: list[str] or None
        :raise lumache.InvalidKindError: If the kind is invalid.
        :return: The ingredients list.
        :rtype: list[str]

        """
        import copy, json
        import numpy as np
        import shapely
        import geopandas as gpd
        import pyrosm

        (grid_cell_id,
         grid_cell_geometry_wkt,
         output_path,
         cntry_poly_wkt,
         google_tiles_path,
         google_buildings_path) = tuple

        # create logger

        print(f'starting to merge osm and google for grid_cell_id: {grid_cell_id}')

        # setup
        grid_cell_geometry = shapely.wkt.loads(grid_cell_geometry_wkt)
        cntry_poly = shapely.wkt.loads(cntry_poly_wkt)
        google_tiles_df = gpd.read_file(google_tiles_path)

        # define output files
        bldgs_in_buffer_geojson_path = os.path.join(output_path,
                                                    f"{grid_cell_id}_geoms.geojson")
        grid_cell_cntry_intrsct_path = os.path.join(output_path, f"{grid_cell_id}_shape.geojson")

        # if the files already exist, skip it. i.e. only run if there's a missing file
        if (os.path.exists(bldgs_in_buffer_geojson_path) and os.path.exists(grid_cell_cntry_intrsct_path)):
            print(f'files already exist: {bldgs_in_buffer_geojson_path}, and {grid_cell_cntry_intrsct_path}')

        else:
            print(f'files do not already exist: {bldgs_in_buffer_geojson_path}, and {grid_cell_cntry_intrsct_path}. '
                  f'Processing now.')
            # first, clip the polygon by the country shape
            grid_cell_cntry_intrsct = cntry_poly.intersection(grid_cell_geometry)

            buffer_radius = 1200  # in m
            buffered_grid_cell_cntry_intrsct = utils_vector.buffer_polygon_using_aep(grid_cell_cntry_intrsct,
                                                                                     buffer_radius)

            # now see what google tiles overlap the buffered grid cell
            intersecting_google_tiles_df = google_tiles_df[google_tiles_df.intersects(buffered_grid_cell_cntry_intrsct)]
            n_intersecting_google_tiles = intersecting_google_tiles_df.shape[0]

            try:
                print('starting to load goog bldgs')

                if intersecting_google_tiles_df.shape[0] == 0:
                    print('No intersecting google tiles in cell')
                    bldgs_in_buffer_gdf = pd.DataFrame()

                # construct dataframe for all buildings that may intersect that grid cell
                for i, google_tile in enumerate(intersecting_google_tiles_df.iterrows()):
                    print(
                        f'---loading google tile {i + 1} of {n_intersecting_google_tiles} for grid cell {grid_cell_id}')

                    google_tile_name = google_tile[1]['tile_id']

                    google_open_buildings_csv_path = os.path.join(google_buildings_path,
                                                                  google_tile_name + '_buildings.csv.gz')
                    bldgs_gdf = utils_vector.load_csv_gdf_with_caching(google_open_buildings_csv_path)

                    if i == 0:
                        all_bldgs_gdf = copy.deepcopy(bldgs_gdf)
                    else:
                        all_bldgs_gdf = all_bldgs_gdf.append(bldgs_gdf)

                bldgs_in_buffer_gdf = all_bldgs_gdf[all_bldgs_gdf.intersects(buffered_grid_cell_cntry_intrsct)]

                del all_bldgs_gdf
                # before starting anything, add a new column to google open buildings data
                n_goog_bldgs_in_buffer = bldgs_in_buffer_gdf.shape[0]

                bldgs_in_buffer_gdf = bldgs_in_buffer_gdf.assign(
                    origin=np.repeat('Google Open Buildings', n_goog_bldgs_in_buffer)
                )
                print('done with google bldgs for now')
            except:
                print('Failed to load google bldgs in cell')

            print('trying to write buildings buffer to file')

            try:
                # write buildings within the buffer to csv:
                bldgs_in_buffer_gdf.to_file(bldgs_in_buffer_geojson_path, driver='GeoJSON')

                # write the shape of interest to geojson:
                gpd.GeoSeries(grid_cell_cntry_intrsct).to_file(grid_cell_cntry_intrsct_path, driver='GeoJSON')
                print('wrote buildings buffer to file')
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

            output_path = os.path.join(os.environ.get("PROJECT_DATA_HDD"), country_code, self.__class__.__name__)
            os.makedirs(output_path, exist_ok=True)
            existing_files = os.listdir(output_path)

            # load all countries shapes
            countries_shapes_df = gpd.read_file(self.args['countries_shapes_path'])
            grid_cells_df = gpd.read_file(self.args['grid_cells_path'])

            # identify country shapely polygon of interest
            cntry_poly = countries_shapes_df[countries_shapes_df['ISO3'] == country_code]['geometry'].iloc[0]
            cntry_poly_wkt = cntry_poly.wkt

            # find all grid cells for which the country intersects
            intersecting_grid_cells_df = grid_cells_df[grid_cells_df.intersects(cntry_poly)]
            n_intersecting_grid_cells = intersecting_grid_cells_df.shape[0]
            print('computed all intersecting grid cells')

            # for each little 0.25 deg grid cell, buffer it, and construct dataframe for all buildings
            # that may intersect that grid cell

            for i, grid_cell in enumerate(intersecting_grid_cells_df.iterrows()):
                grid_cell_id = grid_cell[1]['id']
                grid_cell_geometry_wkt = grid_cell[1]['geometry'].wkt

                if not f'{grid_cell_id}_completed.txt' in existing_files:
                    par_list.append((grid_cell_id,
                                     grid_cell_geometry_wkt,
                                     output_path,
                                     cntry_poly_wkt,
                                     self.args['google_tiles_path'],
                                     self.args['google_buildings_path']))

                    completed_path = os.path.join(output_path, f"{grid_cell_id}_completed.txt")
                    self.outputs[completed_path] = 'x'

        with Pool(self.args['clip_par_pool_size']) as p:
            print(p.map(self.extract_google, par_list, chunksize=2))

    def clip_bldgs_parallel(self):
        if self.outputs_exist():
            print("BuildingClipper: all expected clipped building geojsons exist. Skipping clipping.")
            return
        self.process()
        self.save_metadata()
