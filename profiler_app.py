# coding=utf-8
from flask import Flask, render_template, request, Response, send_file, make_response, jsonify
from numba import jit
import numpy as np
from markupsafe import escape
import os
import matplotlib
import io as IO

import base64
import geopandas as gpd
import shutil
import json
import pandas as pd
import altair as alt
import zipfile
from scipy.interpolate import interp1d as interpolate
#from basic_wrap import requires_auth
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from quickchi_functions import chicalc, getstream, getstream_elev, getchi, get_upstream
from flask_caching import Cache
from shapely.geometry import LineString, Point
from matplotlib.backends.backend_agg import FigureCanvasAgg
from waitress import serve

matplotlib.use('agg')


center = (-30, 130)  # where does initial load?
app = Flask(__name__)
config = {
    "DEBUG": False,  # some Flask specific configs
    "CACHE_TYPE": "filesystem",  # Flask-Caching related configs
    'CACHE_DIR': './templates/cache',
    "CACHE_DEFAULT_TIMEOUT": 300
}

# Data dirs for each continent
basedir = '/media/data1/'
#basedir = '/Volumes/Samsung_T5/'

eudir = basedir + 'eu/'
nadir = basedir + 'na/'
sadir = basedir + 'sa/'
audir = basedir + 'au/'
afdir = basedir + 'af/'
marsdir = basedir+ 'mars/'

# latlon ranges for each continent (top, left, bottom, right)
eurange = (60.0, -14.0, -10, 180)
sarange = (15.0, -93.0, -56, -32)
narange = (60.0, -145.0, 5.0, -52.00)
aurange = (-10.0, 112.0, -56.0, 180.0)
afrange = (40, -19.0, -35, 55)
marsrange = (90, -180, -90, 180)
ranges = [sarange, narange, aurange, afrange, eurange] #Order matters, shouldn't be modified
dirs = [sadir, nadir, audir, afdir, eudir]

app.config.from_mapping(config)
cache = Cache(app)



def getz(dir1, ny, code, smooth=-1):
    """
    Get and store elevation values along stream
    :param dir1:
    :param ny:
    :param code:
    :param smooth: smoothing wl in pixel
    :return:
    """
    
    zfilt = cache.get('zfilt{}'.format(code))#Does it already exist?
    suffx2 = cache.get('suffx2{}'.format(code))
    if zfilt is None:
        #If filtered z values don't exist already, does z exist atleast? 
        zi = cache.get('z{}'.format(code))
        if zi is None:
            strm = cache.get('strm{}'.format(code))
            Is = strm % ny
            Js = np.int64(strm / ny)
            dem_s = np.load(dir1 + 'others/dem{}.npy'.format(suffx2), mmap_mode='r')
            zi = dem_s[ Is, Js].copy()
            zi[zi<-30] = -30

        if smooth > 10:
            smooth = 10
        if smooth == -1: # The default
            smooth = len(zi) / 500 # default
            
        #Otherwise we smooth using user prescribed value
        if smooth > 0:
            zfilt = gaussian_filter1d(zi, np.max([smooth, 1]))
        else:
            zfilt = zi.copy()
        if not (len(zfilt) == len(zi)):
            zfilt = zi.copy()
        cache.set('zfilt{}'.format(code), zfilt)
    return zfilt


def getA(dir1, ny, code, dx):
    """
    Get and cache the accumulation (linear) values along stream
    :param dir1: directory containing files
    :param ny: y dimensions
    :param code: code for caching
    :param dx: x resolution at this latitude
    :return: accumulation (linear) along stream
    """

    suffx = ''
    d8 = cache.get('d8{}'.format(code))

    if d8:
        suffx = '2'
    A = cache.get('acc{}'.format(code))
    if A is None:
        strm = cache.get('strm{}'.format(code))
        Is = strm % ny
        Js = np.int64(strm / ny)

        acc = np.load(dir1 + 'others/acc{}.npy'.format(suffx), mmap_mode='r')
        A = np.float64(acc[Is, Js] * 92.6 * dx)
        cache.set('acc{}'.format(code), A)
    return A


@app.route("/profiler")
# @requires_auth
def main():
    """
    main page
    :return: rendered page
    """
    code = np.random.randint(int(1e7))  # Identifier per user / better way too do this ???
    nuser = cache.get('nuser')
    if nuser is None:
        nuser = 0
    if nuser > 50:
        return "Sorry, too many users at the moment, try again in a few minutes"
    else:
        cache.set('nuser',nuser)
        return (
            render_template("main.html", data=str([[0, 0], [.1, .1]]), code=code, lat1=43, lon1=20, z=5, mainpage=1, err='',
                            dist='0', zfilt='0', n=1, maxdist=0, elevl = -9999))


@app.route('/downstream')
# @requires_auth
def downstream():
    """
    get downstream values for elevation plots
    :return: elevations and locations of stream rendered on main page
    """
    #Get the get variables
    longitude = request.args.get('longitude', type=float)
    latitude = request.args.get('latitude', type=float)
    if longitude < -180:
        longitude +=360
    if longitude > 180: 
        longitude -=360
        
    code = np.random.randint(int(1e7))# We don't expect many users at any given time, so this should be sufficiently random to generate a unique code
    
    zoom = request.args.get('zoom', type=int)
    d8 = request.args.get('d8',type = int)
    mars = request.args.get('user_wants_to_leave_earth',type = int)
    dem2 = request.args.get('dem2',type = int)
    suffx2 = ''
    if dem2:
        suffx2 = '2'
    cache.set('suffx2{}'.format(code),suffx2)
    ## We want to restrict usage if too many queries to not interrupt our users
    nuser = cache.get('nuser')
    if nuser is None:
        nuser=0
    nuser+=1
    cache.set('nuser',nuser)
    
    suffx = ''
    if d8:
        suffx = '2'
    elev = request.args.get('elev', type=int)
    smooth = request.args.get('smooth', type=float)
    
    #The directory dir1 depends on which DEM is being used
    dir1 = None
    for i in range(5):
        upper = ranges[i][0]
        lower = ranges[i][2]
        left = ranges[i][1]
        right = ranges[i][3]

        if i < 3:  #Every continent except for europe/ africa can be described as a box w / little overlap

            if left <= longitude < right:
                if lower <= latitude < upper:
                    dir1 = dirs[i]
                    ubound, lbound = (upper, left)
                    break
        elif i == 3:
            # We have to be careful about overlap between africa and europe 
            P = Point(longitude, latitude)
            g = gpd.read_file('af_bound')
            if g.contains(P)[0]:
                dir1 = afdir
                ubound, lbound = (upper, left)
                break
        elif i == 4:
            dir1 = eudir
            ubound, lbound = (upper, left)
            
    if mars == 1: #Maybe eventually ... 
        dir1 = marsdir
        ubound, lbound = (90, -180)
        
    #Load the DEM and receiver grids using mmap
    dem_s = np.load(dir1 + 'others/dem{}.npy'.format(suffx2), mmap_mode='r')
    stackrx = np.load(dir1 + 'rs/stack_rx{}.npy'.format(suffx), mmap_mode='r')
    stackry = np.load(dir1 + 'rs/stack_ry{}.npy'.format(suffx), mmap_mode='r')

    #
    ny, nx = np.shape(stackrx)
    dy = 92.6
    dx = np.cos(np.abs(latitude) * np.pi / 180) * (1852 / 60) * 3 #Equation for dx depending on the latitude
    if mars:
        res = 1200 * ( 92.6/200)
    else:
        res = 1200
        
    #Starting latitude and longitude
    il = int((ubound - latitude) * res)
    jl = int((longitude - lbound) * res)
    
    if mars:#maybe eventually
        dy = 200
        dx *= 200/92.6
    
    
    # If for some reason elevation didn't get set... 
    z=[]
    if elev is None:
        strm = getstream(il, jl, stackrx, stackry,smooth=smooth)
    else:
        print('1')
        strm, z = getstream_elev(il, jl, stackrx, stackry, dem_s, elev=elev)
        z[z<-30] = -30
    if len(strm) > 1:
        if len(z)>0:
            cache.set('z{}'.format(code), z)

        
        jo = np.float32(strm / ny) * 1 / res + lbound
        io = ubound - np.float32(strm % ny) * 1 / res
        dist = np.cumsum(np.append(np.zeros(1), np.sqrt(((np.float64(io[:-1]) - np.float64(io[1:])) * dy * 1.2) ** 2 + (
                    (np.float64(jo[:-1]) - np.float64(jo[1:])) * dx * 1.2) ** 2)))
        #                 #if len(jo) > 100:
        #                     #depfact = len(jo)/50
        locations = np.array(list(zip(io, jo))).tolist()
        data1 = locations
        
        ## Store data in the cache
        cache.set('dir{}'.format(code), dir1)
        cache.set('latitude{}'.format(code), latitude)
        cache.set('strm{}'.format(code), strm)
        cache.set('dist{}'.format(code), dist)
        cache.set('ny{}'.format(code), ny)
        cache.set('ubound{}'.format(code), ubound)
        cache.set('lbound{}'.format(code), lbound)
        cache.set('d8{}'.format(code),d8)
        
        ## Z should be stored at this point, zfilt will apply the smoothing
        zfilt = getz(dir1, ny, code, smooth=smooth)
        maxz = zfilt[0]
        
        #Interpolate to 1000 datapoints for the d3 plot
        distn = interpolate(np.arange(0, len(dist)), dist)(np.linspace(0.001, float(len(dist)) - 1.001, num=1000))
        dx = distn[1] - distn[0]
        zfiltn = interpolate(dist, zfilt)(distn)

        #Requires list for d3 plot
        zdata = np.array(list(zip(distn, zfiltn))).tolist()

    else: #If there's no stream there ...
        data1 = 0
        zdata = np.zeros(1)
        maxz = 0
        dx = 0
        dist = np.zeros(1)
        err = 'Sorry, no stream data found'
        return render_template("main.html", data=str(data1), code=code, lon1=longitude, lat1=latitude, z=zoom,
                               mainpage=1, err='', maxz=maxz, maxdist=np.max(dist), zdata=zdata, dist=dist.tolist(),
                               dx=dx, elevl = -9999)
        
    minz = np.min(z)
    if smooth > 10:
        err = 'Smoothing was too high, reduced to 10'
    return render_template("main.html", data=str(data1), code=code, lon1=longitude, lat1=latitude, z=zoom, mainpage=0,
                           err='', maxz=maxz, minz = np.min(zfiltn), maxdist=np.max(dist), zdata=zdata, dist=dist.tolist(), dx=dx, elevl = elev)


@app.route('/upstream')
# @requires_auth
def upstream():
    """
    not used yet ...
    :return:
    """
    acc = np.load(accloc, mmap_mode='r')
    stackix = np.load(ixloc, mmap_mode='r')
    stackiy = np.load(iyloc, mmap_mode='r')
    recy = np.load(ryloc, mmap_mode='r')
    recx = np.load(rxloc, mmap_mode='r')
    # f = rio.open('/Volumes/T7/merged_asia.tif')
    ubound, lbound = (60.0, -14.0)
    ny, nx = np.shape(stackix)

    longitude = request.args.get('longitude', type=float)
    latitude = request.args.get('latitude', type=float)

    dy = 92.6
    dx = np.cos(np.abs(latitude) * np.pi / 180) * (1852 / 60) * 3

    zoom = request.args.get('zoom', type=float)

    # m.add_layer(Marker(location=kwargs.get('coordinates')))
    il = int((ubound - latitude) * 1200)
    jl = int((longitude - lbound) * 1200)
    strm = get_upstream(il, jl, stackix, stackiy, recy, recx, acc)
    # ny, nx = np.shape(stackr)
    # if len(strm) > 1:
    jo = np.float32(strm / ny) * 1 / 1200 + lbound
    io = ubound - np.float32(strm % ny) * 1 / 1200
    dist = np.cumsum(
        np.append(np.zeros(1), np.sqrt(((io[:-1] - io[1:]) * 1 * dy) ** 2 + ((jo[:-1] - jo[1:]) * 1 * dx) ** 2)))
    #                 #if len(jo) > 100:
    #                     #depfact = len(jo)/50
    locations = np.array(list(zip(io, jo))).tolist()

    data1 = locations
    io = 0  # images(dist, strm,ny,ubound,lbound,dem_s,acc)
    return render_template("main.html", data=str(data1), code=io, lon1=longitude, lat1=latitude, z=zoom, zfilt=zfilt,
                           dist=dist)


@app.route('/elevplot')
# @requires_auth
def elev_plot():
    """
    :return: Return elevation values and distances for dist-elevation plot in d3
    """
    code = request.args.get('code', type=int)
    # plt.plot()
    plt.xlabel('Distance (km)')
    plt.ylabel('Elevation (m)')
    out = IO.BytesIO()
    plt.savefig(out)
    io1 = base64.b64encode(out.getvalue()).decode('utf-8')

    return render_template("im.html", im=io1, code=code)


@app.route('/chiplot')
# @requires_auth
def chiplot():
    """
    :return: chi - elevation interactive plot
    """
    ## Get the cached data and get variabless
    code = request.args.get('code', type=int)
    dir1 = cache.get('dir{}'.format(code))
    latitude = cache.get('latitude{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    dx = np.cos(np.abs(latitude) * np.pi / 180) * (1852 / 60) * 3
    dist = cache.get('dist{}'.format(code))
    zfilt = getz(dir1, ny, code)
    d8 = cache.get('d8{}'.format(code))
    A = getA(dir1, ny, code, dx)
    
    
    #Distance and z values must be interpolated 
    distn = interpolate(np.arange(1, len(dist)+1), dist)(np.linspace(1, float(len(dist)), num=1000))
    zfiltn = interpolate(dist, zfilt)(distn)
    An = interpolate(dist, A)(distn)
    chi = chicalc(A, dist, .45, U=1.0)
    
    ## Calculate and interpolate chi for various values of theta to be compared
    p = pd.DataFrame() 
    for theta in np.arange(.25, .75, .1):
        chi2 = chicalc(A, dist, theta, U=1.0) 
        chi1 = interpolate(np.arange(0,len(chi2)), chi2)(np.linspace(0,float(len(chi2))-1.01,num=1000))
        zfiltn = interpolate(chi2, zfilt[:-1])(chi1)
        p2 = pd.DataFrame({'Z': zfiltn, 'χ': chi1, 'θ': np.round(np.zeros(len(chi1)) + theta, 2)})
        p = p.append(p2)

    plt.xlabel('chi (m)')
    plt.ylabel('Elevation (m)')

    cache.set('chi{}'.format(code), chi)
    
    #Altair plot
    alt.data_transformers.enable('default', max_rows=1000000)
    brush = alt.selection_interval(encodings=['x'])
    slider = alt.binding_range(min=.25, max=.65, step=.1)
    select_theta = alt.selection_single(name="theta value", fields=['θ'],
                                        bind=slider, init={'θ': .45})

    chart1 = alt.Chart().mark_line().encode(
        x=alt.X('χ:Q',scale=alt.Scale(zero=False)),
        y=alt.Y('Z:Q',scale=alt.Scale(zero=False)),
    ).properties(
        width=600,
        height=500
    ).add_selection(brush
                    ).add_selection(select_theta
                                    ).transform_filter(select_theta)
    chart2 = alt.Chart().mark_line().encode(
        x=alt.X('χ:Q',scale=alt.Scale(zero=False)),
        y=alt.Y('Z:Q',scale=alt.Scale(zero=False)),
    ).properties(
        width=600,
        height=500
    ).transform_filter(
        brush).transform_filter(select_theta)
    chart = alt.vconcat(
        chart1,
        chart2,
        data=p,
        title="Select data to analyze in the top panel, use the slider at the bottom to modify theta"
    )


    io2 = chart.to_html()
    
    #Just return the altair plot as html code
    return render_template("im.html", code=code, chart=io2)  # ,im = io1)


@app.route('/slopeareaplot')
# @requires_auth
def slopeareaplot():
    """
    Needs some work...
    """
    code = request.args.get('code', type=int)
    dir1 = cache.get('dir{}'.format(code))
    ny = cache.get('ny{}'.format(code))

    dist = cache.get('dist{}'.format(code))
    zfilt = getz(dir1, ny, code)

    # plt.plot()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    S = np.log10((np.diff(zfilt) / -np.diff(dist * 1000)))
    A = np.log10(A[1:])
    plt.xlabel('log(Area (sq. m))')
    plt.ylabel('log(Slope (m/m))')
    out = IO.BytesIO()
    plt.savefig(out)
    io1 = base64.b64encode(out.getvalue()).decode('utf-8')
    return render_template("im.html", im=io1, code=code)


@app.route('/get_elev')
# @requires_auth
def get_elev():
    """
    :return:     Get elevation to download
    """
    code = request.args.get('code', type=int)
    dir1 = cache.get('dir{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    zfilt = getz(dir1, ny, code)
    data1 = zfilt.tolist()

    if len(data1) == 0:
        return render_template("download.html", data='No Data')
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='elev', code=code,
                           json=0)


@app.route('/get_dist')
# @requires_auth
def get_dist():
    """
    :return: Distance to download
    """
    code = request.args.get('code', type=int)
    dist = cache.get('dist{}'.format(code))
    if dist is None:
        return render_template("download.html", data='No Data')
    data1 = dist.tolist()
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='dist', code=code,
                           json=0)


@app.route('/get_acc')
# @requires_auth
def get_acc():
    """
    :return: Acc to download
    """
    code = request.args.get('code', type=int)

    latitude = cache.get('latitude{}'.format(code))

    dir1 = cache.get('dir{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    dx = np.cos(np.abs(latitude) * np.pi / 180) * (1852 / 60) * 3
    d8 = cache.get('d8{}'.format(code))

    A = getA(dir1, ny, code, dx)
    if A is None:
        return render_template("download.html", data=str('No Data'))
    data1 = A.tolist()
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='acc', code=code,
                           json=0)

@app.route('/get_chi')
# @requires_auth
def get_chi():
    """
    :return: chi values to download
    """
    code = request.args.get('code', type=int)

    chi = cache.get('chi{}'.format(code))

    if chi is None:
        return "No Data yet - go back and generate a chi plot first!"
    data1 = chi.tolist()
    return render_template("download.html", data=str(data1).replace('[', '').replace(']', ''), name='χ', code=code,
                           json=0)


@app.route('/get_shp')
# @requires_auth
def get_shp():
    """
    :return: geojson - a bit misleading but name is descriptive
    """
    code = request.args.get('code', type=int)
    strm = cache.get('strm{}'.format(code))
    ny = cache.get('ny{}'.format(code))
    lbound = cache.get('lbound{}'.format(code))
    ubound = cache.get('ubound{}'.format(code))

    jo = np.float32(strm / ny) * 1 / 1200 + lbound
    io = ubound - np.float32(strm % ny) * 1 / 1200
    IJ = list(zip(jo, io))

    L = LineString(IJ)
    g = gpd.GeoDataFrame(geometry=[L])
    data1 = g.to_json()
    return Response(data1,
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment;filename={}.geojson'.format(code)})

if __name__ == "__main__":
    #app.run()# host='0.0.0.0', port=13211)
    serve(app, host='0.0.0.0', port=13211)
