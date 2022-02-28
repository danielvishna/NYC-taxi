import pandas as pd
import shapefile
import urllib.request
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import random

#data visualisation
import plotly.express as plt_express
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from shapely.geometry import Polygon, Point
import geopandas as gpd
import seaborn as sns
sns.set()


def download_yellow_trip_data(year, month_range):
    '''
    Download yellow trip data from the given month range in the given year
    :param year: year
    :param month_range: range of months
    :return: csv of the requested data
    '''
    for month in month_range:
        url = "https://s3.amazonaws.com/nyc-tlc/trip+data/" + \
              f"yellow_tripdata_{year}" + \
              "-{0:0=2d}.csv".format(month), \
              f"nyc.{year}" +\
              "-{0:0=2d}.csv".format(month)
        print(url)
        urllib.request.urlretrieve(url[0], url[1])

    return combine_csv(year, month_range)


def load_location_data():
    """
    loads shapefile for New York taxi zones.
    :return: list of zones represented as Polygons, dictionary of taxi zones.
    """
    print("Load location data")
    sf = shapefile.Reader("NYC Taxi Zones/geo_export_04b244fc-db55-4ac1-a1c8-dd804f5f8a13.shp")
    zone_dict = dict()
    polygon_zone = []
    for record in sf.shapeRecords():
        zone_code = record.record[2]
        shape = Polygon(record.shape.points)
        zone_dict[record.record[2]] = {'name': record.record[5],
                                       'center': shape.centroid}
        polygon_zone.append({"name": zone_code, "shape": shape})
    return polygon_zone, zone_dict


def combine_csv(year, month_range):
    """
    Preprocesses the data and combines csv from year and month range.
    :param year: year
    :param month_range: range of months
    :return: combined csv file name, dictionary of taxi zones.
    """
    all_files = [f"nyc.{year}" + "-{0:0=2d}.csv".format(month) for month in tqdm(month_range, desc='Create csvs name')]
    all_csv = [pd.read_csv(f) for f in tqdm(all_files, desc='Read csvs')]
    all_csv = [df.rename(columns={c: c.replace(' ', '_') for c in df.columns}) for df in tqdm(all_csv, desc='Rename columns')]

    print("Concatenate csvs")
    df = pd.concat(all_csv)
    print(df.shape)
    df = df.sample(frac=.007, replace=False)
    print(df.shape)
    polygon_zone, zone_dict = load_location_data()
    print("Add columns to csv")
    df = add_columns(df, polygon_zone)
    print("Remove outliers from csv")
    df = remove_outliers(df)

    file_name = f"combined_csv_{year}_{month_range.start}_{month_range.stop - 1}_{df.shape[0]}.csv"
    df.to_csv(file_name, index=False)
    return file_name, zone_dict


def add_columns(df, polygon_zone):
    '''
    Preprocesses: add columns to the df
    :param df: data fame
    :param polygon_zone: list of zones represented as Polygons
    :return: df with the wanted columns
    '''
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
    '''
    Preprocesses: remove outliers points.
    :param df: data frame
    :return: date frame without the outliers points.
    '''
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
        (df.pickup_longitude < -73.6983) & (df.pickup_longitude > -74.2) &
        (df.pickup_latitude < 40.9181) & (df.pickup_latitude > 40.5422) &
        (df.dropoff_longitude < -73.6983) & (df.dropoff_longitude > -74.0880) &
        (df.dropoff_latitude < 40.9181) & (df.dropoff_latitude > 40.5422) &
        # remove PU and DO zone code that not recognize in the NYC taxi zone
        (df.pickup_zone_code != 264) & (df.pickup_zone_code != 265) &
        (df.dropoff_zone_code != 264) & (df.dropoff_zone_code != 265)
    ]

    return df


def add_location(df: pd.DataFrame, polygon_zone):
    '''
    Add taxi zones to the df.
    :param df: df
    :param polygon_zone: list of zones represented as Polygons
    :return: df with the taxi zones.
    '''
    def set_location(geom_point):
        pickup_point = Point(geom_point[0], geom_point[1])
        dropoff_point = Point(geom_point[2], geom_point[3])
        pickup_zone = None
        dropoff_zone = None
        for zone in polygon_zone:
            if pickup_zone is not None and dropoff_zone is not None:
                return pickup_zone, dropoff_zone
            if pickup_zone is None and pickup_point.within(zone['shape']):
                pickup_zone = zone['name']
            if dropoff_zone is None and dropoff_point.within(zone['shape']):
                dropoff_zone = zone['name']

    temp = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]\
        .apply(set_location, axis=1, result_type='expand')

    df['pickup_zone_code'], df['dropoff_zone_code'] = temp[0], temp[1]
    return df


def make_2D_clustering(df, n_clusters, bbox, day=-1):
    '''
    Make 2D clustering from the df according to the geographic location by the hour.
    :param df: df
    :param n_clusters: number of clusters
    :param bbox: boundary box
    :param day: option to specify day of the week
    '''
    cluster_df = df[["pickup_longitude", "pickup_latitude", "pickup_hour",
                     "day_of_week_in_number"]]

    if day != -1:
        cluster_df = cluster_df[cluster_df["day_of_week_in_number"] == day].drop(["day_of_week_in_number"], axis=1)
    else:
        cluster_df.drop(["day_of_week_in_number"], axis=1, inplace=True)

    labels_list, centroids_list = [], []

    for hour in tqdm(range(0, 24), desc='2D clustering by pickup hour'):
        temp = cluster_df[cluster_df.pickup_hour == hour]

        est = KMeans(n_clusters=n_clusters, init='random')
        est = est.fit(temp)

        label = est.labels_
        centroids = est.cluster_centers_

        labels_list.append(label)
        centroids_list.append(centroids)

        # plot 3D clustering
        plt.figure()

        plt.xlim([bbox[0], bbox[1]])
        plt.ylim([bbox[2], bbox[3]])

        plt.title(f'Clustering rides by pickup hour (Hour: {hour})')
        plt.xlabel('longitude')
        plt.ylabel('latitude')

        plt.scatter(temp['pickup_longitude'], temp['pickup_latitude'],
                    c=label, s=0.001, alpha=0.1)

        centroids_x = [x for x, _, _ in centroids]
        centroids_y = [y for _, y, _ in centroids]

        plt.scatter(centroids_x, centroids_y, s=3, marker='X')

        # plt.savefig(f'plots/clustering_by_hour/Clustering rides by pickup hour (Hour: {hour}).png')
        plt.show()


def make_3D_clustering(df, n_clusters, plot=False, day=-1):
    '''
    Make 3d clustering according to the geographic location and the pickup time
    :param df: df
    :param n_clusters: number of clusters
    :param plot: option to plot the results
    :param day: option to specify day of the week
    :return: the centroids from the cluster.
    '''
    cluster_df = get_cluster_df(df, plot=plot, day=day)

    if day != -1:
        cluster_df = cluster_df[cluster_df["day_of_week_in_number"] == day].drop(["day_of_week_in_number"], axis=1)
    else:
        cluster_df.drop(["day_of_week_in_number"], axis=1, inplace=True)

    label, centroids = clustering_normal(cluster_df, n_clusters)
    if plot:
        plot_clustering(cluster_df, label, centroids)
    return centroids


def get_cluster_df(df, plot=False, day=-1):
    '''extact the data needed for the 3D clustering'''
    cluster_df = df[["pickup_longitude", "pickup_latitude", "pickup_time",
                     "day_of_week_in_number"]]
    if plot:
        plot_pu_by_location_and_time(cluster_df, day=day)
    return cluster_df


# clustering with different axis units
def clustering_normal(temp, n_clusters):
    '''clustering by normalize the axes'''
    normal = [84.2899424, 111.1780341, 0.1]
    temp = temp * normal

    est = KMeans(n_clusters=n_clusters, init='random')
    est = est.fit(temp)

    label = est.labels_
    centroids = est.cluster_centers_
    centroids /= normal

    return label, centroids


def make_neighbors_matrix(df):
    '''
    Make neighbors matrix according to the pickup zone and dropoff zone.
    :param df: df
    :return: neighbors matrix
    '''
    group = df.groupby(['pickup_zone_code', 'dropoff_zone_code'], sort=True,
                       as_index=False)
    temp = group.size()
    index = temp[['pickup_zone_code', 'dropoff_zone_code']] - 1

    matrix = np.zeros(shape=(263, 263))
    matrix[index['pickup_zone_code'].astype(int),
           index['dropoff_zone_code'].astype(int)] = temp['size']
    return matrix


def node_importance(neighbors_matrix: np.ndarray, beta=0.8, epsilon=0.0001):
    '''
    Implements node importance algorithm for the taxi zones according to
    pick-up and dropoff zones
    :param neighbors_matrix: neighbours matrix constructed from the data
    :param beta: chance of teleporting
    :param epsilon: stop condition
    :return: node importance vector
    '''
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


def add_zone_importance(df, zone_importance):
    '''adds zone importance results to the data'''
    def find_importance(zones_code):
        return zone_importance[int(zones_code[0]) - 1], \
               zone_importance[int(zones_code[1]) - 1]

    temp = df[['pickup_zone_code', 'dropoff_zone_code']]\
        .apply(find_importance, axis=1, result_type='expand')

    df['pickup_zone_importance'], df['dropoff_zone_importance'] = temp[0], temp[1]
    return df


def make_bbox(df):
    '''finds bounding box for the location of rides in the data'''
    min_geo = df[['pickup_longitude', 'pickup_latitude']].min() - 0.05
    max_geo = df[['pickup_longitude', 'pickup_latitude']].max() + 0.05
    return (min_geo[0], max_geo[0], min_geo[1], max_geo[1])


def plot_duration_by_(df, subtitle=""):
    '''Plots different connections between the duration and other data variables'''
    sns.catplot(x='passenger_count', y='trip_duration', data=df, kind="strip")\
        .set(
        title=f'Trip duration by passenger count ({subtitle})',
        xlabel="passenger count",
        ylabel="taxi trip duration (in minutes)"
    )
    plt.show()

    sns.lineplot(x='pickup_hour', y='trip_duration', data=df)\
        .set(
        title=f'Average trip duration by pickup hour ({subtitle})',
        xlabel="pickup hour",
        ylabel="average taxi trip duration (in minutes)"
    )
    plt.show()

    sns.lineplot(x='day_of_week_in_number', y='trip_duration', data=df)\
        .set(
        title=f'Average trip duration by day of week ({subtitle})',
        xlabel="day of week (as number)",
        ylabel="average taxi trip duration (in minutes)"
    )
    plt.show()


def plot_pu_by_location_and_time(df, day=-1):
    '''Plots 3D map of pick up locations according to the pick up time'''
    index_list = np.random.randint(0, df.shape[0], int(0.5 * df.shape[0]))
    points = df.iloc[index_list.tolist(), :]
    if day == -1:
        fig = plt_express.scatter_3d(points,
                                     x="pickup_longitude", y="pickup_latitude", z="pickup_time",
                                     color="day_of_week_in_number")
    else:
        fig = plt_express.scatter_3d(points[points["day_of_week_in_number"] == day],
                                     x="pickup_longitude", y="pickup_latitude", z="pickup_time",
                                     color="day_of_week_in_number")
    fig.update_traces(marker=dict(size=1), selector=dict(mode='markers'))
    fig.update_layout(
        scene=dict(
            zaxis=dict(
                ticktext=[str(int(i // 60)) + ":00" for i in
                          range(0, 1401, 60)],
                tickvals=[i for i in range(0, 1401, 60)]
            )
        )
    )
    fig.show()


def plot_zone(df, bbox):
    '''Plots map of pickup locations colored by the zone number'''
    # # Interactive plot
    # fig = plt_express.scatter(df, x="pickup_longitude", y="pickup_latitude",
    #                           color="pickup_zone_code")
    # fig.update_xaxes(range=[bbox[0], bbox[1]])
    # fig.update_yaxes(range=[bbox[2], bbox[3]])

    fig = plt.figure()
    plt.scatter(df['pickup_longitude'], df['pickup_latitude'],
                c=df['pickup_zone_code'], s=0.005)
    plt.xlim([bbox[0], bbox[1]])
    plt.ylim([bbox[2], bbox[3]])
    plt.title('Rides location (colored by taxi zone code)')
    plt.xlabel('longitude')
    plt.ylabel('latitude')

    fig.show()


def plot_zone_importance(zone_dict, node_importance):
    '''plots heat map of taxi zones by importance and bar plot of top taxi zones'''
    long, lat, name, score = [], [], [], []
    for zone_code, zone_properties in zone_dict.items():
        name.append(zone_properties['name'])
        long.append(zone_properties['center'].xy[0][0])
        lat.append(zone_properties['center'].xy[1][0])
        score.append(node_importance[int(zone_code) - 1])

    temp = pd.DataFrame()
    temp['long'] = long
    temp['lat'] = lat
    temp['name'] = name
    temp['score'] = score
    max_score = temp['score'].max()
    min_score = temp['score'].min()
    temp['score'] = temp['score'].apply(lambda x: ((x - min_score) /
                                                   (max_score - min_score)) * 10)

    temp.sort_values(by=['score'], inplace=True, ascending=False)

    # plot heatmap.
    fig, ax = plt.subplots(1, 1)
    NYC_map = gpd.read_file("NYC Taxi Zones/geo_export_04b244fc-db55-4ac1-a1c8-dd804f5f8a13.shp")
    NYC_map['score'] = temp['score']
    NYC_map.plot(column='score', ax=ax, legend=True,
                 legend_kwds={'label': "Zone Importance"}, cmap='OrRd')
    plt.title('Heatmap of taxi zones by zone importance')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.show()

    # plot best zones by zone importance.
    best_zones = temp[:20]
    plt.figure(figsize=(10, 10))
    plt.title('Top 20 taxi zones by zone importance')
    plt.ylabel('zone importance (0-10)')
    plt.xlabel('zone name')
    plt.bar(best_zones['name'], best_zones['score'])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def plot_clustering(points, labels, centroid):
    '''Plots 3d map of clustering results and location of centroids'''
    index_list = np.random.randint(0, points.shape[0], int(0.5 * points.shape[0]))
    points = points.iloc[index_list.tolist(), :]
    labels = labels[index_list]
    fig = plt_express.scatter_3d(points,
                                 x="pickup_longitude", y="pickup_latitude", z="pickup_time",
                                 color=labels,
                                 labels={
                                     "pickup_longitude": "pickup longitude",
                                     "pickup_latitude": "pickup latitude",
                                     "pickup_time": "pickup time"
                                 },
                                 title="3D clustering of taxi rides by Geo location and pickup time"
                                 )

    fig.update_layout(
        scene=dict(
            zaxis=dict(
                ticktext=[str(int(i // 60)) + ":00" for i in range(0, 1401, 60)],
                tickvals=[i for i in range(0, 1401, 60)]
            )
        )
    )

    fig.update_traces(marker=dict(size=2, opacity=0.5), selector=dict(mode='markers'))

    clock_time = np.apply_along_axis(lambda x: str(int(x[0] // 60)) + ":" + str(int(x[0] % 60)), axis=0,
                                     arr=centroid[:, 2].reshape((1, -1)))

    fig.add_trace(go.Scatter3d(x=centroid[:, 0], y=centroid[:, 1], z=centroid[:, 2],
                               text=clock_time, mode='markers',
                               marker=dict(symbol='x', size=2, color='red')))

    for i in range(len(centroid)):
        fig.add_trace(spheres(x=centroid[:, 0][i], y=centroid[:, 1][i],
                              z=centroid[:, 2][i], clr='#000080'))

    # fig.write_html('test.html')
    fig.show()


def spheres(x, y, z, clr):
    '''Plots sphere around given x,y,z point'''
    # Set points. First, do angles
    theta, phi = np.mgrid[0:2 * np.pi:100 * 1j, 0:np.pi:100 * 1j]

    # Set up coordinates for points on the sphere
    x0 = x + 0.011863812 * (np.cos(theta) * np.sin(phi))
    y0 = y + 0.008994582 * (np.sin(theta) * np.sin(phi))
    z0 = z + 10 * np.cos(phi)

    # Set up trace
    trace = go.Surface(x=x0, y=y0, z=z0, colorscale=[[0, clr], [1, clr]],
                       opacity=0.4)
    trace.update(showscale=False)
    return trace


def plot_zone_importance_evaluation(best_10_zones, best_10_zones_name,
                                    optimal_importance, random_walk, title, y_label):
    '''Plots the result of our evaluation of the zone importance method'''
    plt.figure(figsize=(10, 10))
    w = 0.3
    plt.bar(np.arange(len(best_10_zones)) - w, optimal_importance,
            width=w,
            color='b', align='center', label="optimal importance")
    plt.bar(np.arange(len(best_10_zones)), random_walk, width=w,
            color='g',
            align='center', label='random walk')
    plt.xticks(range(len(best_10_zones)), best_10_zones_name)
    plt.legend()
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('zone name')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def evaluate_point(zone_code, minutes_in_day, df: pd.DataFrame, random=False):
    '''evaluates worth of given location for a taxi axd part of
    zone importance evaluation'''
    df = df[df.pickup_time >= minutes_in_day]
    continue_ride = False
    time_wasted = 0
    while not continue_ride:
        if minutes_in_day > 1200:
            return 0, time_wasted, 0

        all_rides_in_zone = df[(df.pickup_zone_code == zone_code) &
                               (df.pickup_time <= minutes_in_day + 1)]
        if all_rides_in_zone.empty:
            minutes_in_day += 1
            time_wasted += 1
            continue

        if flipCoin(1 - (0.7 ** all_rides_in_zone.shape[0])):
            continue_ride = True
        else:
            minutes_in_day += 1
            time_wasted +=1

    if random:
        ride = all_rides_in_zone.sample()
    else:
        max_score = all_rides_in_zone['dropoff_zone_importance'].max()
        all_rides_in_zone = all_rides_in_zone[all_rides_in_zone.dropoff_zone_importance == max_score]
        ride = all_rides_in_zone.sample()

    price, next_time, ride_number = evaluate_point(ride['dropoff_zone_code'].values[0],
                                      ride['dropoff_time'].values[0], df, random)
    return price + ride['total_amount'].values[0], next_time + time_wasted, ride_number + 1


def flipCoin(p):
    '''Flips a coin with a given probability'''
    r = random.random()
    return r < p


def evaluate_zone_importance(df, zone_importance, zone_dict):
    '''Evaluates our zone importance method'''
    best_10_zones = np.argsort(zone_importance)[-10:][::-1] + 1
    best_10_zones_name = [zone_dict[i]['name'] for i in best_10_zones]

    optimal_importance_price, optimal_importance_time_wasted, optimal_importance_ride_number = [], [], []
    random_walk_price, random_walk_time_wasted, random_walk_ride_number = [], [], []
    for zone in tqdm(best_10_zones, desc='evaluate zone importance'):
        price, time_wasted, ride_number = evaluate_point(zone, 420, df, random=False)
        optimal_importance_price.append(price)
        optimal_importance_time_wasted.append(time_wasted)
        optimal_importance_ride_number.append(ride_number)
        price, time_wasted, ride_number = evaluate_point(zone, 420, df, random=True)
        random_walk_price.append(price)
        random_walk_time_wasted.append(time_wasted)
        random_walk_ride_number.append(ride_number)

    plot_zone_importance_evaluation(best_10_zones, best_10_zones_name,
                                    optimal_importance_price, random_walk_price,
                                    title='Comparing random walk to optimal importance from best 10 zone importance',
                                    y_label='total taxi revenue')
    plot_zone_importance_evaluation(best_10_zones, best_10_zones_name,
                                    optimal_importance_time_wasted, random_walk_time_wasted,
                                    title='Comparing random walk to optimal importance from best 10 zone importance',
                                    y_label='total time wasted (seconds)')
    plot_zone_importance_evaluation(best_10_zones, best_10_zones_name,
                                    optimal_importance_ride_number, random_walk_ride_number,
                                    title='Comparing random walk to optimal importance from best 10 zone importance',
                                    y_label='total number of rides')


def evaluate_centroids(df, centroids):
    '''Evaluate given centroid from the clustering'''
    all_day_rides = df.shape[0]
    number_of_rides = 0
    for long, lat, time in tqdm(centroids, desc='evaluate centroid point'):
        rides = df[(df.pickup_longitude >= long - 0.007) & (df.pickup_longitude <= long + 0.007) &
                   (df.pickup_latitude >= lat - 0.007) & (df.pickup_latitude <= lat + 0.007) &
                   (df.pickup_time >= time - 10) & (df.pickup_time <= time + 10)]

        number_of_rides += rides.shape[0]

        df = df[(df.pickup_longitude < long - 0.007) | (df.pickup_longitude > long + 0.007) |
                   (df.pickup_latitude < lat - 0.007) | (df.pickup_latitude > lat + 0.007) |
                   (df.pickup_time < time - 10) | (df.pickup_time > time + 10)]

    return number_of_rides/all_day_rides * 100


def evaluate_clustering(df, centroids, bbox):
    '''Evaluate clustering method implemented'''
    centroids_rides_percentage = evaluate_centroids(df, centroids)

    test_centroids = np.empty(shape=(len(centroids), 3))
    test_centroids[:, 0] = np.random.uniform(bbox[0], bbox[1], size=len(centroids))
    test_centroids[:, 1] = np.random.uniform(bbox[2], bbox[3], size=len(centroids))
    test_centroids[:, 2] = np.random.uniform(0, 1400, size=len(centroids))

    test_rides_percentage = evaluate_centroids(df, test_centroids)

    print(centroids_rides_percentage)
    print(test_rides_percentage)
    return centroids_rides_percentage, test_rides_percentage


def evaluate_clustering_number(df, test_df, n_clusters_range: range, bbox):
    '''evaluate clustering by number of clusters'''
    centroids_percentage_list, test_percentage_list = [], []
    for n_clusters in tqdm(n_clusters_range, desc='evaluate clustering number'):
        centroids = make_3D_clustering(df, n_clusters, plot=False)
        centroids_rides_percentage, test_rides_percentage = evaluate_clustering(test_df, centroids, bbox)
        centroids_percentage_list.append(centroids_rides_percentage)
        test_percentage_list.append(test_rides_percentage)

    plt.figure()
    plt.title('Percentage of rides the centroids "catch" by number of clusters')
    plt.xlabel('number of clusters')
    plt.ylabel('percentage of rides the centroids "catch"')
    plt.plot(list(n_clusters_range), centroids_percentage_list, label='cluster centroids')
    plt.plot(list(n_clusters_range), test_percentage_list, label='test centroids')
    plt.legend()
    plt.show()


def daily_rides(df):
    '''
    plot the number of the daily drives in NYC by given data frame
    '''
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_short_date'] = df['tpep_pickup_datetime'].dt.date
    fig, ax = plt.subplots(figsize=(7, 4.8))
    unique, counts = np.unique(df['pickup_short_date'], return_counts=True)
    ax.plot(unique, counts)
    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.fill_between(unique, 0, counts)
    plt.xlabel('Date')
    plt.ylabel('Number of rides')
    plt.title("Number of daily taxi rides")
    plt.show()


if __name__ == '__main__':
    year = 2016
    month_range = range(1, 5)
    test_month_range = range(5, 7)

    month_range = range(1, 3)
    test_month_range = range(3, 4)

    # # Before download the data
    # file_name, zone_dict = download_yellow_trip_data(year, month_range)
    # test_file_name, _ = download_yellow_trip_data(year, test_month_range)

    # # After download the data
    file_name, zone_dict = combine_csv(year, month_range)
    test_file_name, _ = combine_csv(year, test_month_range)

    # After combine csv
    # file_name = "combined_csv_2016_1_6_8418541.csv"
    # test_file_name = "combined_csv_2016_1_6_328806.csv"
    df = pd.read_csv(file_name)
    test_df = pd.read_csv(test_file_name)
    polygon_zone, zone_dict = load_location_data()

    bbox = make_bbox(df)

    plot_duration_by_(df, subtitle="all rides")
    airport_df = df[
        (df.pickup_zone_code == 138) | (df.pickup_zone_code == 132) | (
                    df.pickup_zone_code == 1)]
    plot_duration_by_(airport_df, subtitle="only rides leaving airports")
    not_airport_df = df[
        (df.pickup_zone_code != 138) & (df.pickup_zone_code != 132) & (
                    df.pickup_zone_code != 1)]
    plot_duration_by_(not_airport_df,
                      subtitle="without rides leaving airports")

    make_2D_clustering(df, 50, bbox)
    centroids = make_3D_clustering(df, 100, plot=True)

    centroids_rides_percentage, test_rides_percentage = evaluate_clustering(test_df, centroids, bbox)
    evaluate_clustering_number(df, test_df, range(20, 101, 10), bbox)

    matrix = make_neighbors_matrix(df)
    zone_importance = node_importance(matrix)
    df = add_zone_importance(df, zone_importance)

    plot_zone(df, bbox)
    plot_zone_importance(zone_dict, zone_importance)

    test_df = add_zone_importance(test_df, zone_importance)
    evaluate_zone_importance(test_df, zone_importance, zone_dict)
