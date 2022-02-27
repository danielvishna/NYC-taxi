# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import plotly.graph_objects
import plotly.graph_objs as go
import shapefile as shp
import geoplot as gplt
import geojson
import shapefile
from descartes import PolygonPatch
# import mpl_toolkits
from fiona.crs import from_string
from geopandas import GeoDataFrame
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import geoplot.crs as gcrs
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import seaborn as sns
import matplotlib.dates as mdates
import plotly.express as plt_express
from sklearn.cluster import KMeans

def download_location_data():
    print("Download location data")
    # urllib.request.urlretrieve("https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip",
    #                            "taxi_zones.zip")
    # with zipfile.ZipFile("taxi_zones.zip","r") as zip_ref:
    #     zip_ref.extractall("./shape")

    sf = shapefile.Reader("C:\\Users\\Daniel\\PycharmProjects\\DS_project\\NYC Taxi Zones\\geo_export_04b244fc-db55-4ac1-a1c8-dd804f5f8a13.shp")
    zone_dict = dict()
    polygon_zone = []
    for record in sf.shapeRecords():
        name = record.record[2]
        shape = Polygon(record.shape.points)
        zone_dict[record.record[2]] = {'name': record.record[5], 'center': shape.centroid}
        polygon_zone.append({"name": name, "shape": shape})
    # fields_name = [field[0] for field in sf.fields[1:]]
    # shp_dic = dict(zip(fields_name, list(range(len(fields_name)))))
    # attributes = sf.records()
    # shp_attr = [dict(zip(fields_name, attr)) for attr in attributes]
    #
    # df_loc = pd.DataFrame(shp_attr).join(get_lat_lon(sf, shp_dic).set_index("OBJECTID"), on="OBJECTID")
    # df_loc.to_csv("location_csv.csv", index=False)
    # return df_loc
    # save_polygon_zone(polygon_zone)
    return polygon_zone, zone_dict

def add_columns(df, polygon_zone):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['trip_duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['trip_duration'] = df['trip_duration'].astype('timedelta64[m]')
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.hour
    df['pickup_time'] = df['tpep_pickup_datetime'].dt.hour * 60 + \
                        df['tpep_pickup_datetime'].dt.minute
    df['dropoff_time'] = df['tpep_dropoff_datetime'].dt.hour * 60 + \
                         df['tpep_dropoff_datetime'].dt.minute
    df['day_of_week_in_number'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['day_of_week_in_number'] = df['day_of_week_in_number'].apply(
        lambda x: ((x + 1) % 7) + 1)
    df = add_location(df, polygon_zone)
    return df

def remove_outliers(df):
    # df = df.dropna(axis=0)
    df.dropna(axis=0, inplace=True)
    df = df[
        # remove total amount <= 0 and total amount >= 1000
        (df.total_amount > 0) &
        (df.total_amount < 1000) &
        # remove passenger <= 0 and passenger >= 7
        (df.passenger_count > 0) &
        (df.passenger_count < 7) &
        # remove trip distance <= 0 and trip distance > 100
        (df.trip_distance > 0) &
        (df.trip_distance <= 100) &
        # remove trip duration <= 0 and trip distance >= 180 (3 hours)
        (df.trip_duration > 0) &
        (df.trip_duration < 180) &
        # remove PU and DO location id that not in NYC taxi zone
        (df.pickup_longitude < -73.6983) & (df.pickup_longitude > -74.0880) &
        (df.pickup_latitude < 40.9181) & (df.pickup_latitude > 40.5422) &
        (df.dropoff_longitude < -73.6983) & (df.dropoff_longitude > -74.0880) &
        (df.dropoff_latitude < 40.9181) & (df.dropoff_latitude > 40.5422) &
        # remove PU and DO zone code that not recognize in the NYC taxi zone
        (df.pickup_zone_code != 264) & (df.pickup_zone_code != 265) &
        (df.dropoff_zone_code != 264) & (df.dropoff_zone_code != 265)
    ]

    # #TODO payment method other than cash and credit card
    # data = data[(data['payment_type'] != 3) & (data['payment_type'] != 4) & (
    #             data['payment_type'] != 5) & (data['payment_type'] != 6)]

    return df

def add_location(df: pd.DataFrame, polygon_zone):
    def set_location(geom_point):
        point = Point(geom_point[0], geom_point[1])
        for zone in polygon_zone:
            if point.within(zone['shape']):
                return zone['name']
    df['pickup_zone_code'] = df[['pickup_longitude', 'pickup_latitude']]\
        .apply(set_location, axis=1)
    df['dropoff_zone_code'] = df[['dropoff_longitude', 'dropoff_latitude']]\
        .apply(set_location, axis=1)

    return df
def lod_data():
    df = pd.read_csv("try.csv")
    polygon_zone, zone_dict = download_location_data()
    print("Add columns to csv")
    df = add_columns(df, polygon_zone)
    print("Remove outliers from csv")
    df = remove_outliers(df)
    df.to_csv("combined_csv1.csv", index=False)
    return df


def plot_duration_by_(df, title=""):
    p = sns.catplot(x='passenger_count', y='trip_duration', data=df, kind="strip")
    p.fig.suptitle(title)
    plt.show()

    p = sns.lineplot(x='pickup_hour', y='trip_duration', data=df)
    p.set_title(title)
    plt.show()

    p = sns.lineplot(x='day_of_week_in_number', y='trip_duration', data=df)
    p.set_title(title)
    plt.show()


def make_clustering(df, n_clusters, plot=False, day=-1):
    cluster_df = get_cluster_df(df, plot=plot, day=day)

    if day != -1:
        cluster_df = cluster_df[cluster_df["day_of_week_in_number"] == day].drop(["day_of_week_in_number"], axis=1)
    else:
        cluster_df.drop(["day_of_week_in_number"], axis=1, inplace=True)

    label, centroids = clustering_normal(cluster_df, n_clusters)
    plot_clustering(cluster_df, label, centroids)

def daily_rides(df):
    '''
    plot the number of the daily drives in NYC by given data frame
    '''
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    df['pickup_short_date'] = df['tpep_pickup_datetime'].dt.date
    fig, ax = plt.subplots()
    unique, counts = np.unique(df['pickup_short_date'], return_counts=True)
    ax.plot(unique, counts)
    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.fill_between(unique, 0, counts)
    plt.show()

def plot_picup(df):
    NYC_map = gpd.read_file("NYC Taxi Zones\\geo_export_04b244fc-db55-4ac1-a1c8-dd804f5f8a13.shp")
    geometry = [Point(xy) for xy in zip(df['pickup_longitude'], df['pickup_latitude'])]
    gdf = GeoDataFrame(df, geometry=geometry)
    gdf.plot(ax=NYC_map.plot(figsize=(10, 6)), marker='o', color='red', markersize=3)
    plt.show()







def make_bbox(df):
    min_geo = df[['pickup_longitude', 'pickup_latitude']].min() - 0.05
    max_geo = df[['pickup_longitude', 'pickup_latitude']].max() + 0.05
    return (min_geo[0], max_geo[0], min_geo[1], max_geo[1])


def get_cluster_df(df, plot=False, day=-1):
    cluster_df = df[["pickup_longitude", "pickup_latitude", "pickup_time",
                     "day_of_week_in_number"]]
    if plot:
        plot_pu_by_location_and_time(cluster_df, day=day)
    return cluster_df

def plot_pu_by_location_and_time(df, day=-1):
    if day == -1:
        fig = plt_express.scatter_3d(df,
                                     x="pickup_longitude", y="pickup_latitude", z="pickup_time",
                                     color="day_of_week_in_number")
    else:
        fig = plt_express.scatter_3d(df[df["day_of_week_in_number"] == day],
                                     x="pickup_longitude", y="pickup_latitude", z="pickup_time",
                                     color="day_of_week_in_number")
    fig.update_traces(marker=dict(size=1), selector=dict(mode='markers'))
    fig.show()


def plot_zone(df, bbox):
    NYC_map = gpd.read_file("NYC_Map.geojson")
    fig = plt_express.choropleth_mapbox(df, geojson=NYC_map, locations="pickup_longitude",
                             color='pickup_zone_code',
                             center={"lat": 40.75, "lon": -73.9},
                             mapbox_style="open-street-map",
                             zoom=8.5)
    fig.add_trace(go.Scatter(x=df["pickup_longitude"], y=df["pickup_latitude"]))#,
                              #color=df["pickup_zone_code"]))
    fig.show()


# clustering with different axis units
def clustering_normal(temp, n_clusters):
    # normal = [1640.42, 1640.42, 1]
    normal = [84.2899424, 111.1780341, 0.1]
    temp = temp * normal

    est = KMeans(n_clusters=n_clusters, init='random')#todo: with randomes
    est = est.fit(temp)

    label = est.labels_
    centroids = est.cluster_centers_
    centroids /= normal

    return label, centroids




def plot_clustering(points, labels, centroid):
    fig = plt_express.scatter_3d(points,
                                 x="pickup_longitude", y="pickup_latitude", z="pickup_time",
                                 color=labels)

    fig.update_traces(marker=dict(size=2, opacity=0.5), selector=dict(mode='markers'))

    clock_time = np.apply_along_axis(lambda x: str(int(x[0] // 60)) + ":" + str(int(x[0] % 60)), axis=0,
                                     arr=centroid[:, 2].reshape((1, -1)))

    fig.add_trace(go.Scatter3d(x=centroid[:, 0], y=centroid[:, 1], z=centroid[:, 2],
                               text=clock_time, mode='markers',
                               marker=dict(symbol='x', size=3, color='red')))

    for i in range(len(centroid)):
        fig.add_trace(spheres(x=centroid[:, 0][i], y=centroid[:, 1][i],
                              z=centroid[:, 2][i], clr='#000080'))
    fig.show()
    fig.write_html("Clustering_rides_pickup_hour.html")

def spheres(x, y, z, clr):
    # Set points. First, do angles
    theta, phi = np.mgrid[0:2 * np.pi:30 * 1j, 0:np.pi:20j]

    # Set up coordinates for points on the sphere
    x0 = x + 0.011863812 * (np.cos(theta) * np.sin(phi))
    y0 = y + 0.008994582 * (np.sin(theta) * np.sin(phi))
    z0 = z + 10 * np.cos(phi)

    # Set up trace
    trace = go.Surface(x=x0, y=y0, z=z0, colorscale=[[0, clr], [1, clr]],
                       opacity=0.4)
    trace.update(showscale=False)
    return trace


def make_neighbors_matrix(df):
    group = df.groupby(['pickup_zone_code', 'dropoff_zone_code'], sort=True,
                       as_index=False)
    temp = group.size()
    index = temp[['pickup_zone_code', 'dropoff_zone_code']] - 1

    matrix = np.zeros(shape=(263, 263))
    matrix[index['pickup_zone_code'].astype(int),
           index['dropoff_zone_code'].astype(int)] = temp['size']
    return matrix

def node_importance(neighbors_matrix: np.ndarray, beta=0.8, epsilon=0.0001):
    temp = neighbors_matrix.sum(axis=0)
    m = neighbors_matrix / np.where(temp == 0, 1, temp)
    n = np.full(shape=m.shape, fill_value=1 / m.shape[0])
    a = beta * m + (1 - beta) * n
    r_old = np.full(shape=m.shape[0], fill_value=1 / m.shape[0])
    r_new = a @ r_old
    while any(np.abs(r_new - r_old) > epsilon):
        r_old = r_new
        r_new = a @ r_old
    return r_new

def plot_zone_importance(zone_dict, node_importance, bbox):
    fig, ax = plt.subplots(1, 1)
    NYC_map = gpd.read_file("C:\\Users\\Daniel\\PycharmProjects\\DS_project\\NYC Taxi Zones\\geo_export_04b244fc-db55-4ac1-a1c8-dd804f5f8a13.shp")
    NYC_map['score'] = node_importance



    NYC_map.plot(column='score', ax=ax, legend=True, legend_kwds={'label': "Zone Importance"})# cax=cax, )
    plt.show()
    f, ax = plt.subplots(1, figsize=(6, 6))
    # Base layer with all the areas for the background
    for poly in NYC_map['geometry']:
        NYC_map.plot(ax=ax, facecolor='black', linewidth=0.025)
    # Smallest areas
    smallest = NYC_map.sort('Total').head(10)
    for poly in smallest['geometry']:
       smallest.plot(ax=ax, alpha=1, facecolor='red', linewidth=0)
    ax.set_axis_off()
    f.suptitle('Areas with smallest population')
    plt.axis('equal')
    plt.show()
    # long = [zone['center'].xy[0][0] for zone in zone_dict.values()]
    # lat = [zone['center'].xy[1][0] for zone in zone_dict.values()]
    # zone_name = [zone['name'] for zone in zone_dict.values()]
    # temp = pd.DataFrame()
    # temp['long'] = long
    # temp['lat'] = lat
    # temp['score'] = pd.Series(node_importance)
    # temp['zone_name'] = zone_name
    # fig = plt_express.choropleth_mapbox(temp, geojson=NYC_map, locations="zone_name",
    #                                     color='score',
    #                                     center={"lat": 40.75, "lon": -73.9},
    #                                     # mapbox_style="carto-positron",
    #                                     mapbox_style="open-street-map",
    #                                     zoom=9)

    # fig = plt_express.scatter(temp, x="long", y="lat",
    #                           color="score", size='score')
    # fig.update_xaxes(range=[bbox[0], bbox[1]])
    # fig.update_yaxes(range=[bbox[2], bbox[3]])
    # fig.show()

if __name__ == '__main__':
    # df = lod_data()

    NYC_map = gpd.read_file(
        "C:\\Users\\Daniel\\PycharmProjects\\DS_project\\NYC Taxi Zones\\geo_export_04b244fc-db55-4ac1-a1c8-dd804f5f8a13.shp")

    pnt1 = Point(40.496115395170364, -73.7000090639354)
    pnt2 = Point(40.496115395170364, -74.25559136315209)
    points_df = gpd.GeoDataFrame({'geometry': [pnt1, pnt2]}, crs='EPSG:4326')
    points_df = points_df.to_crs('EPSG:5234')
    points_df2 = points_df.shift()  # We shift the dataframe by 1 to align pnt1 with pnt2
    df = pd.read_csv("combined_csv.csv")
    df = df.sample(n = 2500)
    # plot_picup(df)
    polygon_zone, zone_dict = download_location_data()

    matrix = make_neighbors_matrix(df)
    zone_importance = node_importance(matrix)
    bbox = make_bbox(df)

    # plot_duration_by_(df)
    make_clustering(df, 10, plot=True)
    # plot_zone(df, bbox)
    # plot_zone_importance(zone_dict, zone_importance, bbox)













    # sns.set(style="whitegrid", palette ="pastel", color_codes = True)
    # sns.mpl.rc("figure", figsize = (10, 6))
    # shp_path = "taxi_zones\\taxi_zones.shp"
    # sf = gpd.read_file(shp_path)
    # sf.plot()
    # plt.show()
    # # df = read_shapefile(sf)
    # # comuna = 'Bay Ridge'
    # # com_id = sf.index[sf.zone == comuna].tolist()[0]
    # # plot_map(sf)
    # # plot_map2(25, sf)
    # # comuna_id = [0, 1, 2, 3, 4, 5, 6]
    # # plot_map_fill_multiples_ids("Multiple Shapes",
    # #                             comuna_id, sf, color='r')
    # south = ['Newark Airport', 'Jamaica Bay', 'Alphabet City', "Astoria", 'Astoria Park', 'Auburndale', 'Battery Park City',
    #          'Bay Terrace/Fort Totten', 'Bayside', 'Bedford Park', 'Bellerose', 'Belmont', 'Bensonhurst East', 'Cambria Heights',
    #          'Carroll Gardens', 'Central Harlem', 'Central Park', 'Flushing']
    # data = [100, 2000, 300, 400000, 500, 600, 100, 2000, 300, 400, 500, 600, 100, 2000, 300, 400, 500, 600]
    # print_id = True  # The shape id will be printed
    # color_pallete = 1  # 'Purples'
    # plot_comunas_data(sf, 'South', south, data, color_pallete, print_id)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
