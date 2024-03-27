import sys, os, time, json, math, pickle
from shapely.geometry import shape
import shapely
import pyproj
from functools import partial
import datatable as dt
import geopandas as gpd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
sys.path.append(os.environ.get("PROJECT_ROOT"))



def buffer_polygon_using_aep(geom, radius):
    # create_circle_using_azimuthal_equidistant_projection

    # radius in meters
    lon = geom.centroid.xy[0][0]
    lat = geom.centroid.xy[1][0]

    local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format(
        lat, lon
    )
    wgs84_to_aeqd = partial(
        pyproj.transform,
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
        pyproj.Proj(local_azimuthal_projection),
    )
    aeqd_to_wgs84 = partial(
        pyproj.transform,
        pyproj.Proj(local_azimuthal_projection),
        pyproj.Proj("+proj=longlat +datum=WGS84 +no_defs"),
    )

    geom_transformed = shapely.ops.transform(wgs84_to_aeqd, geom)
    buffer = geom_transformed.buffer(radius)
    # Get the polygon with lat lon coordinates
    buffer_geom = shapely.ops.transform(aeqd_to_wgs84, buffer)

    return buffer_geom


def load_csv_gdf_with_caching(csv_path):
    pickle_path = os.path.splitext(csv_path)[0] + '.p'

    if os.path.exists(pickle_path):

        print(f'loading pickle: {pickle_path}')
        bldgs_gdf = pickle.load(open(pickle_path, "rb"))

    else:

        print(f'reading and saving to pickle: {csv_path}')
        bldgs_gdf = dt.fread(csv_path).to_pandas()
        bldgs_gdf['geometry'] = bldgs_gdf['geometry'].apply(shapely.wkt.loads)
        bldgs_gdf = gpd.GeoDataFrame(bldgs_gdf, geometry='geometry', crs="EPSG:4326")

        bldgs_gdf.to_pickle(pickle_path)

    bldgs_gdf.rename(columns={'full_plus_code': 'origin_id'}, inplace=True)

    return bldgs_gdf


def geom_to_area(geom):
    crs_4326 = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(
        crs_4326,
        pyproj.CRS(proj='aea',
                   lat_1=geom.bounds[1],
                   lat_2=geom.bounds[3]
                   )
    )
    geom_area = shapely.ops.transform(transformer.transform, geom)
    return geom_area.area


def generate_vector_grid(outputGridfn, xmin, xmax, ymin, ymax, gridHeight, gridWidth):
    from math import ceil
    from osgeo import ogr

    # get rows
    rows = ceil((ymax - ymin) / gridHeight)
    # get columns
    cols = ceil((xmax - xmin) / gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax - gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn, geom_type=ogr.wkbPolygon)
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom = ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Close DataSources
    outDataSource.Destroy()