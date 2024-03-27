import os, sys, argparse, copy, json

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))

from litldf.compile_features.google_building_clipper import GoogleBuildingClipper
from litldf.compile_features.microsoft_building_clipper import MicrosoftBuildingClipper
from litldf.compile_features.building_merger import BuildingMerger
from litldf.compile_features.building_counter import BuildingCounter
from litldf.compile_features.lulc_adder import LulcAdder
from litldf.compile_features.ntl_adder import NtlAdder
from litldf.compile_features.ook_adder import OokAdder


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Get shapefile in gridded form for country.')

    parser.add_argument('-ccs', '--country_codes',
                        type=str,
                        default='["RWA"]',
                        help='The ISO country codes for the countries being analyzed')

    parser.add_argument('-pps', '--par_pool_size_all',
                        type=int,
                        default=6,
                        help='The ISO country codes for the countries being analyzed')

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args = args.__dict__
    args['country_codes'] = json.loads(args['country_codes'])

    print(f"starting runs for countries: {args['country_codes']}!")

    # Clip Google buildings
    GoogleBuildingClipper(copy.deepcopy(args),
                          clip_par_pool_size=args['par_pool_size_all'],
                          countries_shapes_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                             'raw',
                                                             'afr_g2014_2013_0',
                                                             'afr_g2014_2013_0.shp'),
                          google_tiles_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                         'raw',
                                                         'tiles',
                                                         'tiles.geojson'),
                          grid_cells_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                       'raw',
                                                       '0.25deg-world-grid',
                                                       '0.25deg-world-grid.geojson'),
                          google_buildings_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                             'raw',
                                                             'polygons_s2_level_4_gzip'),
                          osm_pbf_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                    'raw',
                                                    'planet-220425.osm.pbf')).clip_bldgs_parallel()

    # Clip Microsoft buildings
    MicrosoftBuildingClipper(copy.deepcopy(args),
                             par_pool_size=args['par_pool_size_all'],
                             previous_class_name='GoogleBuildingClipper',
                             countries_shapes_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                                'raw',
                                                                'afr_g2014_2013_0',
                                                                'afr_g2014_2013_0.shp'),
                             microsoft_cells_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                               'raw',
                                                               'msft_bldgs',
                                                               'buildings-coverage_20240111',
                                                               'buildings-coverage.geojson'),
                             microsoft_buildings_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                                                   'raw',
                                                                   'msft_bldgs',
                                                                   'microsoft_bldgs_2024_prioritycntries',
                                                                   'microsoft_bldgs_2024')).clip_bldgs_parallel()

    # Merge buildings
    BuildingMerger(copy.deepcopy(args),
                   par_pool_size=args['par_pool_size_all'],
                   google_class_name='GoogleBuildingClipper',
                   microsoft_class_name='MicrosoftBuildingClipper').merge_bldgs_parallel()

    BuildingCounter(copy.deepcopy(args),
                    par_pool_size=args['par_pool_size_all'],
                    shapes_class_name='GoogleBuildingClipper',
                    previous_class_name='BuildingMerger').count_bldgs_parallel()

    LulcAdder(copy.deepcopy(args),
              par_pool_size=args['par_pool_size_all'],
              previous_class_name='BuildingCounter',
              lulc_utm_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                         'raw',
                                         'LULC',
                                         'lulc2017',
                                         'lc2017'),
              lulc_feature_shares_to_add='["water", "trees", "flooded_veg", "crops", "built_area", "bare_ground", "rangeland"]',
              lulc_classes='{"1": "water", "2": "trees", "4": "flooded_veg", "5": "crops", "7": "built_area", "8": "bare_ground", "9": "snow_ice", "10": "clouds", "11": "rangeland"}',
              lulc_year=2017,
              window_sizes='[1, 11, 51]').add_lulc_parallel()

    NtlAdder(copy.deepcopy(args),
             par_pool_size=args['par_pool_size_all'],
             previous_class_name='LulcAdder',
             ntl_path=os.path.join(os.environ.get("PROJECT_DATA"),
                                   'raw',
                                   'ntl',
                                   'VNL_v2_npp_2018_global_vcmslcfg_c202101211500.average.tif'),
             ntl_year=2018,
             window_sizes='[1, 11, 51]',
             feat_name='ntl').add_ntl_parallel()

    OokAdder(copy.deepcopy(args),
             par_pool_size=args['par_pool_size_all'],
             previous_class_name='NtlAdder',
             ookla_path_1=os.path.join(os.environ.get("PROJECT_DATA"),
                                       'raw',
                                       'ookla',
                                       '2020-01-01_performance_fixed_tiles',
                                       'gps_fixed_tiles.shp'),
             feat_name_1='ookla_fixed',
             ookla_date_1='20200101',
             ookla_path_2=os.path.join(os.environ.get("PROJECT_DATA"),
                                       'raw',
                                       'ookla',
                                       '2020-01-01_performance_mobile_tiles',
                                       'gps_mobile_tiles.shp'),
             feat_name_2='ookla_mobile',
             ookla_date_2='20200101').add_ook_parallel()


if __name__ == '__main__':
    main(sys.argv[1:])
