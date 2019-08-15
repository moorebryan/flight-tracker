from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.spatial import distance
from bokeh.transform import linear_cmap
from bokeh.palettes import inferno
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import directed_hausdorff
import copy
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.embed import components

# connect the app
app = Flask(__name__)

# Make connection to MongoDB Server
MONGODB_URL=''
client = MongoClient(MONGODB_URL)
db = client.datamanager
curs=db.flightTracks.find()
equip_list = db.flightTracks.distinct('equipment')
num_clusters = 4
num_return = 100 # Limit to grabbing this many flights

# Definition returns minimum euclidean distance between provided node and target node
def closest_node(comp_node, nodes):
    # Note nodes is a list of tuples
    node_list = []
    if len(nodes[0])>2:
        for index,node in enumerate(nodes):
            node_list.append((node[0],node[1]))
    else:
        node_list = nodes
    closest_index = distance.cdist([comp_node], node_list).argmin()
#     print(nodes[closest_index])
    return nodes[closest_index]

def read_in_flights(filter_code,lat_lon_range,date):
    # No filtering is filter_code=0
    # Filter by departing from LAX is filter_code=1
    # Filter by arriving to LAX is filter_code=2

    lax_coord = [33.9416, -118.4085]
    # Range of latitudes and longitudes around LAX to include
    lat_min = lax_coord[0] - lat_lon_range
    lat_max = lax_coord[0] + lat_lon_range
    lon_min = lax_coord[1] - lat_lon_range
    lon_max = lax_coord[1] + lat_lon_range

    df = pd.DataFrame()
    
    num_flagged = 0
    
    prev_day = date[:-1] + str(int(date[0])-1)
    next_day = date[:-1] + str(int(date[0])+1)

    if filter_code == 0: # No filtering is filter_code=0
        data=pd.DataFrame(list(db.flightTracks.find({'arrivalAirportFsCode':'LAX',
                                                     "departureDate.dateUtc": { "$gte" : prev_day,
                                                                               "$lt" : next_day }}).limit(num_return)),
                          columns=['_id','departureAirportFsCode','arrivalAirportFsCode','equipment',
                                   'departureDate','positions'])
        start_index = 0
        end_index = -1
    elif filter_code == 1: # Filter by departing from LAX is filter_code=1
        data=pd.DataFrame(list(db.flightTracks.find({'arrivalAirportFsCode':'LAX',
                                                     "departureDate.dateUtc": { "$gte" : prev_day,
                                                                               "$lt" : next_day }}).limit(num_return)),
                          columns=['_id','departureAirportFsCode','arrivalAirportFsCode','equipment',
                                   'departureDate','positions'])
        start_index = -400
        end_index = -1
    elif filter_code == 2: # Filter by arriving to LAX is filter_code=2
        data=pd.DataFrame(list(db.flightTracks.find({'arrivalAirportFsCode':'LAX',
                                                     "departureDate.dateUtc": { "$gte" : prev_day,
                                                                               "$lt" : next_day }}).limit(num_return)),
                          columns=['_id','departureAirportFsCode','arrivalAirportFsCode','equipment',
                                   'departureDate','positions'])
        start_index = 0
        end_index = 400
            
    num_flagged += data.shape[0]
            
    # No go through each flight within this type of equipment
    out = []
    out_dates = []
    skip_list = [True] * data.shape[0]
    # Go through each flight
    for index in range(data.shape[0]):
        
        temp_df = pd.DataFrame(data.positions[index],
                               columns=['lat','lon','altitudeFt','date','course','speedMph',
                                        'vrateMps']).iloc[start_index:end_index]
            
        # Store as list of tuples
        temp_df['points'] = list(zip(temp_df.lat,
                                     temp_df.lon,
                                     temp_df.altitudeFt,
                                     temp_df.date,
                                     temp_df.course))
            
        if len(temp_df)==0:
            skip_list[index] = False
        else:
            # Only include if flight contains points near LAX
            near_node=closest_node(lax_coord, list(zip(temp_df.lat,temp_df.lon)))
            if (near_node[0]<lat_min or near_node[0]>lat_max):
                skip_list[index] = False
            if (near_node[1]<lon_min or near_node[1]>lon_max):
                skip_list[index] = False
            
        # Append list of tuples for particular plane
        out.append(temp_df.points.values)
        out_dates.append(data.departureDate[index]['dateUtc'])
        
    # Add these tuples to the original dataframe and remove the positions column
    data['points'] = out
    data['date'] = out_dates
        
    # Before returning filter out those rows flagged as being skipped
    data = data[skip_list]
        
    data.drop(columns=['positions'],inplace=True)
    df = pd.concat([df, data], axis=0, sort=False)

    df.reset_index(inplace=True)
    return df, num_flagged

def plot_arrivals(df_plot):
    
    # Range of planes to plot
    plot_min=0
    plot_max=df_plot.shape[0]
    
    left = -140
    right = -80
    bottom = 25
    top = 45
    p = figure(title="Ungrouped Flight Trajectories Arriving To LAX",x_range=(left, right), y_range=(bottom, top))
    
    #color_mapper = linear_cmap('index', inferno(256), plot_min, plot_max)
    
    for i in range(plot_min,plot_max):
    
        # Overlay all flights on each other
        temp_df = df_plot.iloc[i-1:i]
        
        # Gather list of latitude and longitude points for each flight separately
        for elem in temp_df.points:
            lat = [item[0] for item in elem]
            lon = [item[1] for item in elem]
            
            index_list = np.ones(len(lat))*i
            source = ColumnDataSource(
                data=dict(lat=lat,
                          lon=lon,
                          index=index_list)
            )
    
            p.circle(x="lon", y="lat", size=2, fill_alpha=0.1, source=source)
            
    p.xaxis.axis_label = 'Longitude'
    p.yaxis.axis_label = 'Latitude'
    
    return p

def crt_traj_lst(traj_data):
    
    window = 1 # 1 means no smoothing
    
    traj_lst = []
    alt_list = []
    for flight_instance in traj_data:
    
        temp_pos_list = []
        
        keep = 1
        for index,point in enumerate(flight_instance):

            temp_pos_list.append([point[0],point[1],point[4]])
            alt_list.append(point[2])
        
        # If valid append to list as valid flight
        if keep == 1 and len(temp_pos_list)>10:
            temp_pos_list = np.asarray(temp_pos_list)
            
            # Smooth before saving
            avg_mask = np.ones(window) / window
            out = []
            for i in list(range(3)):
                y = [elem[i] for elem in temp_pos_list]
                out.append(np.convolve(np.asarray(y), avg_mask, 'same')[1:-1])
            output = np.zeros((len(out[0]),3))
            output[:,0] = out[0]
            output[:,1] = out[1]
            output[:,2] = out[2]
            output = np.array(output)
            
            traj_lst.append(np.vstack(output))
    return traj_lst

def scale_data(traj_lst):
    scaler = StandardScaler()
    
    mat = [0]*3
    row_list = []
    prev_row = 0
    for curr_mat in traj_lst:
        # Keep track of row so we can put standardized values back in original list
        curr_row = prev_row + len(curr_mat)
        row_list.append([prev_row,curr_row])
        prev_row = curr_row
        # Stack all values on top of each other so we can scale
        mat = np.vstack((mat, curr_mat))
        
    # remove initializing row
    mat = np.delete(mat, 0, 0)
    # Scale the data
    mat = scaler.fit_transform(mat)
    
    # Now put the standardized values back into traj_lst
    for curr_index,elem in enumerate(row_list):
        traj_lst[curr_index] = mat[elem[0]:elem[1]]
    return traj_lst

# Calculate distance matrix using hausdorff distance
def hausdorff( u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d

# Calculates distance between each flight and every other flight
def calc_sim_matrix(traj_lst):
    traj_count = len(traj_lst)
    D = np.zeros((traj_count, traj_count))
    
    # This may take a while
    for i in range(traj_count):
        for j in range(i + 1, traj_count):
            distance = hausdorff(traj_lst[i], traj_lst[j])
            D[i, j] = distance
            D[j, i] = distance
    return D

def plot_cluster(traj_lst, cluster_lst):
    #num_clusters = 10
    left = -140
    right = -80
    bottom = 25
    top = 45
    p = figure(title="Clustered Flight Trajectories Arriving To LAX",x_range=(left, right), y_range=(bottom, top))
    color_mapper = linear_cmap('index', inferno(num_clusters), 0, num_clusters)
    
    for traj, cluster in zip(traj_lst, cluster_lst):
        # Gather list of latitude and longitude points for each flight separately
        lat = traj[:,0]
        lon = traj[:,1]
        index_list = np.ones(len(lat))*cluster
        
        source = ColumnDataSource(
                data=dict(lat=lat,
                          lon=lon,
                          index=index_list)
                )
        p.circle(x="lon", y="lat", size=2, fill_alpha=0.1, source=source,
                 color=color_mapper)
        
    p.xaxis.axis_label = 'Longitude'
    p.yaxis.axis_label = 'Latitude'
    
    return p

from bokeh.models import Legend
def crt_cluster_subplots(traj_lst,cluster_lst):
    # Create subplots for all clusters
    left = -123
    right = -113
    bottom = 31
    top = 36

    # First create plot of standard arrival procedures
    df_plot=pd.read_csv('https://raw.githubusercontent.com/moorebryan/flight-tracker/master/data/STAR_df.csv')
    proc_list=[]
    # Convert trajectories from strings to lists
    for index in range(df_plot.shape[0]):
        # cut off [[ at beginning and ]] at the end
        curr_traj = str(df_plot.iloc[index].Trajectory[2:-2])
        split_traj = curr_traj.split('], [')

        new_traj = []
        for point in split_traj:

            # Pull out pieces of each point corresponding to lat,lon,orientation
            lat,lon,orientation = point.split(',')
            # Create list of points, properly formatted
            new_traj.append([float(lat),float(lon),float(orientation)])

        proc_list.append(new_traj)
    plots=[]
    
    for index in range(len(np.unique(cluster_lst))):
        # Create figure for this cluster
        curr_title = 'Cluster '+ str(index)+' Overlayed On Arrival Procedures'
        p = figure(plot_width=550, plot_height=550, title=curr_title,
                   x_range=(left, right), y_range=(bottom, top))
        p.xaxis.axis_label = 'Longitude'
        p.yaxis.axis_label = 'Latitude'
        
        # Create arrival procedures we overlay flights on
        for traj_proc in proc_list:
            traj_proc = np.array(traj_proc)
            p.line(x=traj_proc[:,1], y=traj_proc[:,0], line_dash=[4,4],
                   line_color="black",legend='FAA Procedures')
        
        # Add flights included in current cluster
        for traj_flight, cluster in zip(traj_lst, cluster_lst):
            if cluster==index:
                lat = traj_flight[:,0]
                lon = traj_flight[:,1]
                index_list = np.ones(len(lat))*cluster
                source = ColumnDataSource(
                        data=dict(lat=lat,
                                  lon=lon,
                                  index=index_list)
                        )
                p.line(x="lon", y="lat", source=source,line_color="blue",
                       legend='Airplane Trajectories')
                
        # Append this created plot to list to be embedded in html
        p.legend.location = "bottom_center"
        plots.append(p)
    return plots


@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/analysis')
def analysispage():
    return render_template('analysis.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results', methods=['POST', 'GET'])
def resultspage():
    
    try:
        date = request.form["day_choice"] # date formatted like yyyy-mm-dd
    except:
        return render_template("analysis.html")
    
    filter_code = 2 # For arriving to LAX
    lat_lon_range = 1.0 # Range around LAX for nearest node to include for filtering
    arr_LAX_df,num_flagged = read_in_flights(filter_code,lat_lon_range,date)
    plot1 = plot_arrivals(arr_LAX_df)
    # Embed plot into HTML via Flask Render
    script1, div1 = components(plot1)
    
    traj_lst = crt_traj_lst(arr_LAX_df.points)
    
    unscaled_traj_lst = copy.deepcopy(traj_lst) # Needed for later
    # Scale trajectories
    traj_lst = scale_data(traj_lst)
    # Calculate similiarity matrix
    D = calc_sim_matrix(traj_lst)
    
    # Calculate K-Means Clustering predictions
    # Number of clusters
    kmeans = KMeans(num_clusters)
    # Fitting the input data
    kmeans = kmeans.fit(D)
    # Getting the cluster labels
    cluster_lst = kmeans.predict(D)
    
    plot2 = plot_cluster(unscaled_traj_lst, cluster_lst)
    # Embed plot into HTML via Flask Render
    script2, div2 = components(plot2)
    
    # Save plot with clusters separately
    plot3=crt_cluster_subplots(unscaled_traj_lst,cluster_lst)
    script3, div3 = components(plot3)
    
    return render_template("results.html", script1=script1, div1=div1,
                           script2=script2, div2=div2,
                           script3=script3, div3=div3)

if __name__ == '__main__':
    app.run(port=33507)



