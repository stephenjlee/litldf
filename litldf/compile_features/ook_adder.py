import sys, os
import numpy as np
from multiprocessing import Pool
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from litldf.compile_features.metadata_helper import MetadataLogger

class OokAdder(MetadataLogger):
    ookla_gdf_1 = None
    ookla_gdf_2 = None

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
    def add_ook(tuple):

        (i,
         unique_file_id,
         input_path,
         output_path,
         n_unique_file_ids,
         country_code,
         props) = tuple

        try:

            input_geoms_geojson_path = os.path.join(input_path, f"{unique_file_id}_geoms.geojson")

            bldgs_gdf = gpd.read_file(input_geoms_geojson_path)
            bldgs_gdf["origin_origin_id"] = bldgs_gdf["origin"] + '_' + bldgs_gdf["origin_id"]

            print('loaded bldgs')

            bldgs_gdf['geometry_orig'] = bldgs_gdf['geometry'].copy()
            bldgs_gdf['geometry'] = bldgs_gdf['geometry'].centroid
            # only doing an intersection
            # on points now, to keep from computing for (any? as many?) duplicates
            # Duplicates would occur if more than one pixel (shape) intersects a building.

            for feat_name, ookla_date, ookla_gdf in zip([props.args['feat_name_1'], props.args['feat_name_2']],
                                                        [props.args['ookla_date_1'], props.args['ookla_date_2']],
                                                        [OokAdder.ookla_gdf_1, OokAdder.ookla_gdf_2]):

                bldgs_gdf = bldgs_gdf.sjoin(ookla_gdf, how="left", predicate='intersects')
                print('finished spatial join')

                ookla_columns = ookla_gdf.columns.values
                ookla_columns = np.delete(ookla_columns, np.where(ookla_columns == 'geometry'))
                cols_to_list = ['index', 'quadkey']
                cols_to_mean = np.setdiff1d(ookla_columns, cols_to_list).tolist()
                cols_to_list = [f'{feat_name}_{ookla_date}_{key}' for key in cols_to_list]
                cols_to_mean = [f'{feat_name}_{ookla_date}_{key}' for key in cols_to_mean]
                ookla_columns_rename_mapping = {key: f'{feat_name}_{ookla_date}_{key}' for key in ookla_columns}
                ookla_columns_rename_mapping['index_right'] = f'{feat_name}_{ookla_date}_index'
                bldgs_gdf.rename(columns=ookla_columns_rename_mapping, inplace=True)
                print('finished column renaming')

                # account for duplicates
                unique_ids = bldgs_gdf['origin_origin_id'].values
                unique_unique_ids, counts_unique_ids = np.unique(unique_ids, return_counts=True)
                unique_ids_with_duplicates = unique_unique_ids[counts_unique_ids > 1]

                print(f'found {unique_ids_with_duplicates.size} duplicate unique_ids')
                for j, id in enumerate(unique_ids_with_duplicates):
                    print(f'processing duplicate {j} of {unique_ids_with_duplicates.size}, id {id}')

                    bldgs_overlap_df = bldgs_gdf[bldgs_gdf['origin_origin_id'] == id].copy()
                    proposed_row = bldgs_overlap_df.iloc[0].copy()

                    for col in cols_to_list:
                        col_list = bldgs_overlap_df[col].values.astype(str)
                        col_list_string = ','.join(col_list)
                        proposed_row[col] = col_list_string
                    for col in cols_to_mean:
                        col_list = bldgs_overlap_df[col].values
                        proposed_row[col] = np.nanmean(col_list)

                    print('dropping duplicate rows')
                    for ind in np.unique(bldgs_overlap_df.index.values):
                        bldgs_gdf = bldgs_gdf.drop(ind)

                    print('adding proposed row')
                    bldgs_gdf = pd.concat([bldgs_gdf, pd.DataFrame([proposed_row])], ignore_index=True)

                # fill nan values with 0
                bldgs_gdf = bldgs_gdf.fillna(value={col: 0 for col in cols_to_mean})
                print('fill nan values with 0')

            # switch back geometries and delete points
            bldgs_gdf['geometry'] = bldgs_gdf['geometry_orig'].copy()
            bldgs_gdf.geometry = bldgs_gdf['geometry']
            bldgs_gdf = bldgs_gdf.drop(columns=['geometry_orig'])

            #####################################

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

        print('reading ookla_gdf_1')
        OokAdder.ookla_gdf_1 = gpd.read_file(self.args['ookla_path_1'])
        print('reading ookla_gdf_2')
        OokAdder.ookla_gdf_2 = gpd.read_file(self.args['ookla_path_2'])

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
                    print(f'added {unique_file_id} to the list for generating ook geometries.')
                    par_list.append((i,
                                     unique_file_id,
                                     input_path,
                                     output_path,
                                     n_unique_file_ids,
                                     country_code,
                                     self))

                    self.outputs[completed_path] = 'x'

            with Pool(self.args['par_pool_size']) as p:
                print(p.map(self.add_ook, par_list, chunksize=1))


    def add_ook_parallel(self):
        if self.outputs_exist():
            print("OokAdder: all expected ook added building files exist. Skipping adding.")
            return
        self.process()
        self.save_metadata()
