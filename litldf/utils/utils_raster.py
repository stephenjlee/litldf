import copy
import sys, os
import geopandas as gpd
import rasterio as rio
import numpy as np
import utm

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))


def add_utm_cols_to_latlon_gdf(gdf):
    lons = gdf.geometry.centroid.x.values
    lats = gdf.geometry.centroid.y.values

    zone_numbers = []
    zone_letters = []
    zone_number_letters = []
    for lat, lon in zip(lats, lons):
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        zone_numbers.append(zone_number)
        zone_letters.append(zone_letter)
        zone_number_letters.append(f'{str(zone_number).zfill(2)}{zone_letter}')

    gdf['zone_number'] = zone_numbers
    gdf['zone_letter'] = zone_letters
    gdf['zone_number_letters'] = zone_number_letters

    return gdf


def reproject_coords(src_crs, dst_crs, coords):
    import fiona.transform

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    xs, ys = fiona.transform.transform(src_crs, dst_crs, xs, ys)
    return [[x, y] for x, y in zip(xs, ys)]


def return_window(dataset, lon, lat, N, outfile=None):
    src_crs = 'EPSG:4326'
    dst_crs = dataset.crs.to_proj4()  # '+proj=moll +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m no_defs'
    dst_crs = f"EPSG:{dst_crs[dst_crs.find(':') + 1:]}"
    new_coords = reproject_coords(src_crs, dst_crs, [[lon, lat]])  # new coords in UTM zone meters

    px = new_coords[0][0]
    py = new_coords[0][1]
    iy, ix = dataset.index(px, py)  # pixel coordinates

    # Build an NxN window
    window = rio.windows.Window(ix - N // 2, iy - N // 2, N, N)

    # Read the data in the window
    # clip is a nbands * N * N numpy array
    clip = dataset.read(window=window)

    # You can then write out a new file
    meta = dataset.meta
    meta['width'], meta['height'] = N, N
    meta['transform'] = rio.windows.transform(window, dataset.transform)

    if outfile is not None:
        with rio.open(outfile, 'w', **meta) as dst:
            dst.write(clip)

    value = list(rio.sample.sample_gen(dataset, new_coords))[0][0]

    return clip, value


def get_shares_by_class(values, classes):
    shares = {}

    for key in classes:
        shares[classes[key]] = np.sum(values == int(key)) / values.size

    return shares


def add_class_value_and_local_histogram_to_gdf(bldgs_gdf_filtered,
                                               raster,
                                               year,
                                               classes,
                                               feature_shares_to_add,
                                               feature_base_name,
                                               window_sizes: list = [1, 5, 11]):
    # initialize list of lists for shares to add correspond to each feature
    feature_shares_by_window_size = []
    for _ in window_sizes:
        feature_shares_by_window_size.append({feat: [] for feat in feature_shares_to_add})

    # loop through each building
    for i, row in bldgs_gdf_filtered.iterrows():
        lat = row.latitude
        lon = row.longitude

        # convert single value or clip from window to shares by class
        shares_by_class = []
        for i, window_size in enumerate(window_sizes):
            if window_size == 1:
                _, value = return_window(raster, lon, lat, 1)
                shares_by_class.append(get_shares_by_class(np.array([value]), classes))
            else:
                clip, _ = return_window(raster, lon, lat, window_size)
                shares_by_class.append(get_shares_by_class(clip.flatten(), classes))

        # save share to data structure for saving
        for i, _ in enumerate(window_sizes):
            for key in feature_shares_by_window_size[i]:
                feature_shares_by_window_size[i][key].append(shares_by_class[i][key])

    # add to geodataframe
    for i, window_size in enumerate(window_sizes):
        for key in feature_shares_by_window_size[i]:
            bldgs_gdf_filtered[f'{feature_base_name}{year}_{key}_N{window_size}'] = feature_shares_by_window_size[i][
                key]

    return bldgs_gdf_filtered

def add_feat_value_and_local_means_to_gdf(bldgs_gdf,
                                          raster,
                                          year,
                                          feat_name,
                                          window_sizes: list = [1, 5, 11]):
    # initialize list of lists for shares to add correspond to each feature
    feature_means_by_window_size = []
    for _ in window_sizes:
        feature_means_by_window_size.append({feat_name: []})

    for i, row in bldgs_gdf.iterrows():
        lat = row.latitude
        lon = row.longitude

        # convert single value or clip from window to shares by class
        means = []
        for i, window_size in enumerate(window_sizes):
            if window_size == 1:
                _, value = return_window(raster, lon, lat, 1)
                means.append(value)
            else:
                clip, _ = return_window(raster, lon, lat, window_size)
                means.append(np.mean(clip))

        # save share to data structure for saving
        for i, _ in enumerate(window_sizes):
            for key in feature_means_by_window_size[i]:
                feature_means_by_window_size[i][key].append(means[i])

    # add to geodataframe
    for i, window_size in enumerate(window_sizes):
        for key in feature_means_by_window_size[i]:
            bldgs_gdf[f'{feat_name}{year}_N{window_size}'] = feature_means_by_window_size[i][
                key]

    return bldgs_gdf
