import numpy as np
import xarray as xr
import pandas as pd
import scipy.signal as signal

# Additional helper functions

def rmean(data, n):
    try:
        return pd.Series(data).rolling(window=n, min_periods=1, center=True).mean().values
    except TypeError as error:
        print(error)
        print("Returning initial tab.")
        return data
    
def extend_lon(dataarray, n_lon, lon_name='longitude'):

    if n_lon==0:
        return dataarray
    if lon_name == 'longitude':
        dataslice = xr.DataArray(dataarray.isel(longitude=slice(0,n_lon)))
        dataslice = dataslice.assign_coords(longitude=dataslice.longitude+360)
        return xr.concat([dataarray,dataslice],dim='longitude')
    elif lon_name == 'longitude_1':
        dataslice = xr.DataArray(dataarray.isel(longitude_1=slice(0,n_lon)))
        dataslice = dataslice.assign_coords(longitude_1=dataslice.longitude_1+360)
        return xr.concat([dataarray,dataslice],dim='longitude_1')
    elif lon_name == 'lon':
        dataslice = xr.DataArray(dataarray.isel(lon=slice(0,n_lon)))
        dataslice = dataslice.assign_coords(lon=dataslice.lon+360)
        return xr.concat([dataarray,dataslice],dim='lon')

def cycle_box(lon_min, lon_max, lat_min, lat_max):
    return [[lon_min, lon_min, lon_max, lon_max, lon_min],
            [lat_min, lat_max, lat_max, lat_min, lat_min]]
    
def create_coordinate_edges(coordinates):
    """
    Create bound file for regriding
    :param coordinates: 1D and regular!
    :return:

    """
    step = coordinates[1] - coordinates[0]
    return [coordinates[0] - step / 2 + i * step for i in range(len(coordinates) + 1)]

def cell_area(n_lon, lat1, lat2):
    """
    Area of a cell on a regular lon-lat grid.
    :param n_lon: number of longitude divisions
    :param lat1: bottom of the cell
    :param lat2: top of the cell
    :return:
    """
    r = 6371000
    lat1_rad, lat2_rad = 2 * np.pi * lat1 / 360, 2 * np.pi * lat2 / 360
    return 2 * np.pi * r ** 2 * np.abs(np.sin(lat1_rad) - np.sin(lat2_rad)) / n_lon

def surface_matrix(lon, lat):
    """
    Compute a matrix with all the surfaces values.
    :param lon:
    :param lat:
    :return:
    """
    n_j, n_i = len(lat), len(lon)
    lat_b = create_coordinate_edges(lat)
    surface_matrix = np.zeros((n_j, n_i))
    for i in range(n_i):
        for j in range(n_j):
            surface_matrix[j, i] = cell_area(n_i, lat_b[j], lat_b[j + 1])
    return surface_matrix
    

class ButterLowPass:
    
    def __init__(self, order, fc, fs, mult=1):
        self.order = order
        self.fc = fc * mult
        self.fs = fs
        self.filter = signal.butter(order, fc * mult, 'lp', fs=1, output='sos')
    
    def process(self, data):
        return signal.sosfiltfilt(self.filter, data - np.mean(data)) + np.mean(data)
    
    def plot(self, min_power=None, max_power=None, n_fq=100):
        """
        
        :param min_power:
        :param max_power:
        :param n_fq:
        :return: w : ndarray, The angular frequencies at which `h` was computed.
        :return: h : ndarray, The frequency response.
        """
        nyq = 0.5 * self.fs
        normal_cutoff = self.fc / nyq
        # b: Numerator of a linear filter; a: Denominator of a linear filter
        b, a = signal.butter(self.order, normal_cutoff, btype='low', analog=True)
        if min_power is not None and max_power is not None:
            return signal.freqs(b, a, worN=np.logspace(min_power, max_power, n_fq))
        return signal.freqs(b, a)


# ------------------------------------------------------------ #
# ---------- COLLECTION BOXES AND SPREADING REGIONS ---------- #
# ------------------------------------------------------------ #

# Imported from https://github.com/earyro/mw_protocol

class Grid:
    
    def __init__(self, lat, lon):
        self.lon_center, self.lat_center = np.meshgrid(lon.center, lat.center)
        self.lon_lower, self.lat_lower = np.meshgrid(lon.lower, lat.lower)
        self.lon_upper, self.lat_upper = np.meshgrid(lon.upper, lat.upper)
    
    def area(self):
        """
        Area of grid cell is
        S(i,j) = R * R *(crad * (lon.upper[i] -  lon.lower[i])) *
                (sin(lat.upper[j]) - sin(lat.lower[j]))
        """
        r = 6371000.0  # radius of Earth (m)
        crad = np.pi / 180.0
        area = r * r * (crad * (self.lon_upper - self.lon_lower)) * \
               (np.sin(crad * self.lat_upper) - np.sin(crad * self.lat_lower))
        area_globe = np.sum(area)
        area_globe_true = 4 * np.pi * r * r
        assert abs(area_globe - area_globe_true) <= area_globe_true * 1e-6
        # print "calculated numerical area is",area_globe,',',100*area_globe/area_globe_true,'% arithmetical value'
        area = np.copy(area)
        return area

class LonAxis:
    """Define longitude axis boundaries
    and deal with wrapping around"""
    
    def __init__(self, lon):
        lon_p = np.roll(lon, -1)  # shifted longitude
        lon_p[-1] += 360
        lon_m = np.roll(lon, 1)
        lon_m[0] -= 360
        lon_lower = lon - (lon - lon_m) / 2.0
        lon_upper = lon + (lon_p - lon) / 2.0
        #
        self.center = lon[:]
        self.lower = lon_lower[:]
        self.upper = lon_upper[:]


class LatAxis:
    """Define latitude axis boundaries
    and overwrite pole boundaries"""
    
    def __init__(self, lat):
        lat_p = np.roll(lat, -1)  # shifted
        lat_m = np.roll(lat, 1)
        lat_lower = lat - (lat - lat_m) / 2.0
        lat_upper = lat + (lat_p - lat) / 2.0
        #
        self.center = lat[:]
        self.lower = lat_lower[:]
        self.upper = lat_upper[:]
        self.lower[0] = -90
        self.upper[-1] = 90

class Box:
    
    def __init__(self, latmin, latmax, lonmin, lonmax):
        self.latmin = latmin
        self.latmax = latmax
        if lonmin < 0:
            lonmin += 360.0
        if lonmax < 0:
            lonmax += 360.0
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.cells_in = None
        self.ocean_in = None
        self.nc = None
        self.no = None
        # self.get_mask(grid,mask)
    
    def get_mask(self, grid, mask):
        """Count ocean grid boxes within the area"""
        # define grid arrays
        lons = grid.lon_center[:]
        lats = grid.lat_center[:]
        ocean_boxes = np.logical_not(mask)
        #
        lats_in = np.logical_and(lats < self.latmax, lats > self.latmin)
        lons_in = np.logical_and(lons < self.lonmax, lons > self.lonmin)
        self.cells_in = np.logical_and(lats_in, lons_in)
        self.ocean_in = np.logical_and(self.cells_in, ocean_boxes)
        self.nc = np.sum(self.cells_in)
        self.no = np.sum(self.ocean_in)
    
    def __repr__(self):
        return str(self.no) + ' ocean cells in the box'

    def cycle_box(self):
        return [[self.lonmin, self.lonmin, self.lonmax, self.lonmax, self.lonmin],
                [self.latmin, self.latmax, self.latmax, self.latmin, self.latmin]]


class Region:
    
    def __init__(self, boxes, grid, mask):
        self.boxes = boxes[:]
        self.grid = grid
        self.grid_mask = np.copy(mask)
        self.mask = None
        self.no = None
        self.totalarea = None
        
        self.get_mask()
        self.calc_area()
    
    def get_mask(self):
        """Count ocean grid boxes within the region"""
        # define grid arrays
        ocean_boxes = np.logical_not(self.grid_mask)
        #
        ocean_in = np.zeros(ocean_boxes.shape)  # start with no box
        for box in self.boxes:
            # add cells from each box
            box.get_mask(self.grid, self.grid_mask)
            ocean_in = np.logical_or(ocean_in, box.ocean_in)
        self.mask = np.copy(ocean_in)
        self.no = np.sum(self.mask)
    
    def calc_area(self):
        """ calculate surface of the region"""
        self.totalarea = np.ma.array(self.grid.area(), mask=np.logical_not(self.mask[:])).sum()
    
    def calc_total_flux(self):
        pass
    
    def __repr__(self):
        return str(self.no) + ' ocean cells in the region'


def generate_collection_boxes():
    """
    Method to define the collection boxes.
    :return: Dictionary of collection Boxes.
    """
    
    collection_boxes = dict()
    # Define rivers (automatically defines masks and areas)
    # Start with defining the new regions to put the water
    
    # USA East Coast
    collection_boxes["USECoast1"] = Box(37, 46, -70, -52)
    collection_boxes["USECoast2"] = Box(32, 37, -70, -65)
    collection_boxes["USECoast3"] = Box(28.75, 43, -81, -70)
    collection_boxes["USECoast4"] = Box(40, 46, -52, -48)
    collection_boxes["USECoast5"] = Box(46, 50, -66, -58)
    collection_boxes["USECoast6"] = Box(40, 46, -48, -46)  # New One, only for catching

    # Greenland Arctic
    collection_boxes["GrArc1"] = Box(81, 88, 279.5, 346)
    # North American Arctic
    collection_boxes["NAMArc1"] = Box(78, 86, 271, 279.5)
    collection_boxes["NAMArc2"] = Box(68.75, 86, 246, 271)
    collection_boxes["NAMArc3"] = Box(60, 82, 233, 246)
    collection_boxes["NAMArc4"] = Box(60, 80, 191, 233)
    collection_boxes["NAMArc5"] = Box(55, 68.75, 250, 264.375)  # only for catching the water, not for spreading it
    collection_boxes["NWTerr1"] = Box(55, 60, 235, 246)  # only for catching the water
    collection_boxes["NWTerr2"] = Box(55, 66, 246, 250)  # not for spreading it
    # Great Lakes  # Can decide which spreading box to add this to
    collection_boxes["GrLakes1"] = Box(43, 48.75, -90, -72)  # only for catching the water, not for spreading it
    # Gulf of Mexico
    collection_boxes["GoM1"] = Box(17.7, 28.75, -96.3, -80)
    # East Pacific
    collection_boxes["EPac1"] = Box(50, 60, 191, 215.5)
    collection_boxes["EPac2"] = Box(50, 60, 215.5, 225.5)
    collection_boxes["EPac3"] = Box(38.5, 60, 225.5, 234.5)
    collection_boxes["EPac4"] = Box(33.75, 38.5, 230, 260)
    collection_boxes["EPac5"] = Box(28.5, 33.75, 234.5, 260)
    # Russia Pacific
    collection_boxes["RussPac1"] = Box(58, 68, 178, 191)
    # Labrador Sea & Baffin Bay
    collection_boxes["BafLab1"] = Box(68.75, 80, 275, 317)
    collection_boxes["BafLab2"] = Box(50, 68.75, 294.25, 317)
    collection_boxes["BafLab3"] = Box(46, 50, 305.75, 317)
    collection_boxes["HudBay1"] = Box(48.75, 68.75, 264.375, 294.375)  # only for catching the water
    collection_boxes["HudBay2"] = Box(51, 54, 260, 264.375)  # not for spreading it
    # Atlantic Greenland Iceland
    collection_boxes["AtlGr1"] = Box(58, 71.25, 317, 337.25)
    collection_boxes["AtlGr2"] = Box(62.5, 63.75, 337.25, 339.5)
    # E Greenland & Iceland
    collection_boxes["EGrIce1"] = Box(63.75, 81, 337.25, 346)
    collection_boxes["EGrIce2"] = Box(68.75, 83, 346, 357)
    # E Iceland
    collection_boxes["EIceland1"] = Box(63.75, 68.75, 346, 351)
    # UK Atlantic
    collection_boxes["UKAtl1"] = Box(46, 62.5, 346.75, 360)
    # Eurasian GIN Seas
    collection_boxes["EurGIN1"] = Box(68, 80, 3, 9.5)
    collection_boxes["EurGIN2"] = Box(68, 78, 9.5, 24.375)
    collection_boxes["EurGIN3"] = Box(60, 68, 0, 16)
    collection_boxes["EurGIN4"] = Box(50, 60, 0, 13)
    collection_boxes["EurGIN5"] = Box(66.25, 68, 16, 24.375)
    collection_boxes["EurGIN6"] = Box(68, 80, 0., 3)  # New one, only for catching
    collection_boxes["Baltic1"] = Box(50, 60.0, 13, 30)  # only for catching the water
    collection_boxes["Baltic2"] = Box(60, 66.25, 16, 38)  # not for spreading
    # South Iceland
    collection_boxes["SIceland1"] = Box(60, 63.75, 339.5, 346.75)
    # Siberian Arctic
    collection_boxes["SibArc1"] = Box(68, 82, 173, 191)
    collection_boxes["SibArc2"] = Box(68, 82, 114.5, 173)  # New One
    # Eurasian Arctic
    collection_boxes["EurArc1"] = Box(78, 86, 9.5, 114.5)
    collection_boxes["EurArc2"] = Box(66.25, 78, 24.375, 114.5)
    collection_boxes["EurArc3"] = Box(80, 86, 0, 9)  # New One - only for catching
    # Mediterranean
    collection_boxes["Med1"] = Box(29, 40, 0, 41.5)
    collection_boxes["Med2"] = Box(40, 45, 0, 24)
    collection_boxes["BlckSea1"] = Box(40, 50, 26, 42)  # only for catching the water, not for spreading it
    collection_boxes["CaspSea1"] = Box(35, 50, 46, 55)  # NEW ONE , only for catching
    # Patagonia Atlantic
    collection_boxes["PatAtl1"] = Box(-56.25, -40.0, 290.5, 305)
    # Patagonia Pacific
    collection_boxes["PatPac1"] = Box(-56.25, -36, 282, 290.5)
    collection_boxes["PatPac2"] = Box(-57.5, -56.25, 282, 294.5)
    # New Zealand (South)
    collection_boxes["SNZPac1"] = Box(-47.5, -43.75, 167, 176)
    # New Zealand (North)
    collection_boxes["NNZPac1"] = Box(-43.75, -39, 165, 174.25)
    # Antarctic Ross Sea
    collection_boxes["AARos1"] = Box(-90.0, -68.0, 167.0, 239.0)
    # Antarctic Amundsen Sea
    collection_boxes["AAAmund"] = Box(-90.0, -60.0, 239.0, 297.0)
    # Antarctic Weddell Sea
    collection_boxes["AAWeddell"] = Box(-90.0, -60.0, 297.0, 360.0)
    # Antarctic Riiser-Larson Sea
    collection_boxes["AARiiLar"] = Box(-90.0, -60.0, 0.0, 59)
    # Antarctic Davis Sea
    collection_boxes["AADavis"] = Box(-90.0, -60.0, 59.0, 167.0)

    return collection_boxes


def generate_spreading_regions(cb, um_grid, masked, masked_500m):
    """
    Method to define the collection boxes.
    :param cb: Collection boxes dictionary.
    :param um_grid: Grid of the model.
    :param masked: Land sea mask at the surface.
    :param masked_500m: Land sea mask at 500m.
    :return: List of spreading zones objects.
    """
    
    # Now identify the regions that the water is routed into and spread it over the new larger regions
    us_ecoast = {'name': 'US_East_Coast',
                 'loc': Region([cb["USECoast1"], cb["USECoast2"], cb["USECoast3"], cb["USECoast4"], cb["USECoast5"],
                                cb["USECoast6"], cb["GrLakes1"]], um_grid, masked),
                 'region': Region(
                     [cb["USECoast1"], cb["USECoast2"], cb["USECoast3"], cb["USECoast4"], cb["USECoast4"],
                      cb["USECoast5"]], um_grid,
                     masked_500m)}
    gr_arc = {'name': 'Greenland_Arctic', 'loc': Region([cb["GrArc1"]], um_grid, masked),
              'region': Region([cb["GrArc1"]], um_grid, masked_500m)}
    n_am_arc = {'name': 'N_American_Arctic',
                'loc': Region(
                    [cb["NAMArc1"], cb["NAMArc2"], cb["NAMArc3"], cb["NAMArc4"], cb["NAMArc5"], cb["NWTerr1"],
                     cb["NWTerr2"]], um_grid, masked),
                'region': Region([cb["NAMArc1"], cb["NAMArc2"], cb["NAMArc3"], cb["NAMArc4"]], um_grid, masked_500m)}
    g_o_m = {'name': 'Gulf_of_Mexico', 'loc': Region([cb["GoM1"]], um_grid, masked),
             'region': Region([cb["GoM1"]], um_grid, masked_500m)}
    e_pac = {'name': 'East_Pacific',
             'loc': Region([cb["EPac1"], cb["EPac2"], cb["EPac3"], cb["EPac4"], cb["EPac5"]], um_grid, masked),
             'region': Region([cb["EPac1"], cb["EPac2"], cb["EPac3"], cb["EPac4"], cb["EPac5"]], um_grid,
                              masked_500m)}
    russ_pac = {'name': 'Russia_Pacific', 'loc': Region([cb["RussPac1"]], um_grid, masked),
                'region': Region([cb["RussPac1"]], um_grid, masked_500m)}
    baf_lab = {'name': 'LabradorSea_BaffinBay',
               'loc': Region([cb["BafLab1"], cb["BafLab2"], cb["BafLab3"], cb["HudBay1"], cb["HudBay2"]], um_grid,
                             masked),
               'region': Region([cb["BafLab1"], cb["BafLab2"], cb["BafLab3"]], um_grid, masked_500m)}
    atl_gr = {'name': 'Atlantic_GreenlandIceland', 'loc': Region([cb["AtlGr1"], cb["AtlGr2"]], um_grid, masked),
              'region': Region([cb["AtlGr1"], cb["AtlGr2"]], um_grid, masked_500m)}
    e_gr_ice = {'name': 'EastGreenland_Iceland', 'loc': Region([cb["EGrIce1"], cb["EGrIce2"]], um_grid, masked),
                'region': Region([cb["EGrIce1"], cb["EGrIce2"]], um_grid, masked_500m)}
    e_ice = {'name': 'EastIceland', 'loc': Region([cb["EIceland1"]], um_grid, masked),
             'region': Region([cb["EIceland1"]], um_grid, masked_500m)}
    uk_atl = {'name': 'UK_Atlantic', 'loc': Region([cb["UKAtl1"]], um_grid, masked),
              'region': Region([cb["UKAtl1"]], um_grid, masked_500m)}
    eur_gin = {'name': 'Eurasian_GINSeas', 'loc': Region(
        [cb["EurGIN1"], cb["EurGIN2"], cb["EurGIN3"], cb["EurGIN4"], cb["EurGIN5"], cb["EurGIN6"], cb["Baltic1"],
         cb["Baltic2"]],
        um_grid, masked),
               'region': Region([cb["EurGIN1"], cb["EurGIN2"], cb["EurGIN3"], cb["EurGIN4"], cb["EurGIN5"]], um_grid,
                                masked_500m)}
    s_iceland = {'name': 'South_Iceland', 'loc': Region([cb["SIceland1"]], um_grid, masked),
                 'region': Region([cb["SIceland1"]], um_grid, masked_500m)}
    sib_arc = {'name': 'Siberian_Arctic', 'loc': Region([cb["SibArc1"], cb["SibArc2"]], um_grid, masked),
               'region': Region([cb["SibArc1"]], um_grid, masked_500m)}
    eur_arc = {'name': 'Eurasian_Arctic',
               'loc': Region([cb["EurArc1"], cb["EurArc2"], cb["EurArc3"]], um_grid, masked),
               'region': Region([cb["EurArc1"], cb["EurArc2"]], um_grid, masked_500m)}
    med = {'name': 'Mediterranean',
           'loc': Region([cb["Med1"], cb["Med2"], cb["BlckSea1"], cb["CaspSea1"]], um_grid, masked),
           'region': Region([cb["Med1"], cb["Med2"]], um_grid, masked_500m)}
    pat_atl = {'name': 'Patagonia_Atlantic', 'loc': Region([cb["PatAtl1"]], um_grid, masked),
               'region': Region([cb["PatAtl1"]], um_grid, masked_500m)}
    pat_pac = {'name': 'Patagonia_Pacific', 'loc': Region([cb["PatPac1"], cb["PatPac2"]], um_grid, masked),
               'region': Region([cb["PatPac1"], cb["PatPac2"]], um_grid, masked_500m)}
    nnz_pac = {'name': 'NorthNewZealand_Pacific', 'loc': Region([cb["NNZPac1"]], um_grid, masked),
               'region': Region([cb["NNZPac1"]], um_grid, masked_500m)}
    snz_pac = {'name': 'SouthNewZealand_Pacific', 'loc': Region([cb["SNZPac1"]], um_grid, masked),
               'region': Region([cb["SNZPac1"]], um_grid, masked_500m)}
    aa_ros = {'name': 'Antarctica_RossSea', 'loc': Region([cb["AARos1"]], um_grid, masked),
              'region': Region([cb["AARos1"]], um_grid, masked_500m)}
    aa_amund = {'name': 'Antarctica_AmundsenSea', 'loc': Region([cb["AAAmund"]], um_grid, masked),
                'region': Region([cb["AAAmund"]], um_grid, masked_500m)}
    aa_weddell = {'name': 'Antarctica_WeddellSea', 'loc': Region([cb["AAWeddell"]], um_grid, masked),
                  'region': Region([cb["AAWeddell"]], um_grid, masked_500m)}
    aa_rii_lar = {'name': 'Antarctica_RiiserLarsonSea', 'loc': Region([cb["AARiiLar"]], um_grid, masked),
                  'region': Region([cb["AARiiLar"]], um_grid, masked_500m)}
    aa_davis = {'name': 'Antarctica_DavisSea', 'loc': Region([cb["AADavis"]], um_grid, masked),
                'region': Region([cb["AADavis"]], um_grid, masked_500m)}
    
    return [us_ecoast, gr_arc, n_am_arc, g_o_m, e_pac, russ_pac, baf_lab, atl_gr, e_gr_ice, e_ice, uk_atl, eur_gin,
            s_iceland, eur_arc, sib_arc, med, pat_atl, pat_pac, nnz_pac, snz_pac, aa_ros, aa_amund, aa_weddell,
            aa_rii_lar, aa_davis]
    

def convert_waterfix(wfix, init_mw, surface_matrix):
    """
    Convert the waterfix to m3/S and resize it to fit the waterfix.
    :param wfix: Waterfix.
    :param init_mw: Discharge file to extract the shape from.
    :param surface_matrix: LSM surface matrix.
    :return: 3D converted waterfix [m3/S, t*lat*lon]
    """
    nt, nlat, nlon = init_mw.shape
    d = 1000  # water density
    
    wfix = np.where(np.isnan(wfix), 0, wfix)
    wfix_fnl_flux = wfix / d * surface_matrix
    wfix_fnl_3d = np.resize(wfix_fnl_flux, (nt, nlat, nlon))  # expand the 2D waterfix with time
    
    return wfix_fnl_3d


def remove_waterfix(ds_discharge, ds_wfix):
    """
    Remove waterfix from a discharge dataset.
    :param ds_discharge: Discharge dataset (in m3/s).#
    :param ds_wfix: Discharge dataset (in kg/m2/s).
    :return: A discharge dataset without the waterfix.
    """
    
    print("____ Removing waterfix")

    assert ds_discharge.attrs['waterfix'] is not None, "There is no waterfix on the discharge dataset."

    ds_output = ds_discharge.copy()

    if 'unspecified' in ds_wfix.coords.keys():
        wfix = ds_wfix.field672.isel(unspecified=0).isel(t=0).isel(longitude=slice(None,-2)).values
    elif 'depth' in ds_wfix.coords.keys():
        wfix = ds_wfix.field672.isel(depth=0).isel(t=0).isel(longitude=slice(None,-2)).values
    else:
        wfix = ds_wfix.field672.isel(t=0).isel(longitude=slice(None,-2)).values

    # Converting the waterfix (kg/m2/s1) to discharge unit (m3/s) and make it 3D
    wfix_fnl_3d = convert_waterfix(wfix, ds_discharge.discharge.values, 
                                   surface_matrix(ds_discharge.longitude.values, ds_discharge.latitude.values))
    
    ds_output.discharge.values = ds_discharge.discharge.values - wfix_fnl_3d
    ds_output.attrs['waterfix'] = None

    return ds_output


def create_discharge_ts(ds_discharge, ds_lsm, rmean=None, details='low', units=None):
    """
    Create the discharge series for plot_discharge_ts.
    :param ds_discharge: Dataset with discharge to plot.
    :param ds_lsm: Dataset with land_sea_mask.
    :param rmean: running mean years
    :param details: clustering complexity, low shows all the zones.
    :param units: force writing units.
    :return: Discharge time series in Sv.
    """

    print("__ Creating discharge time series")

    lat, lon = LatAxis(ds_discharge.latitude.values), LonAxis(ds_discharge.longitude.values)
    umgrid = Grid(lat, lon)
    
    masked = np.copy(ds_lsm.lsm.values)  # land mask True (1) on land
    depthm = np.ma.masked_less(ds_lsm.depthdepth.values, 500.0)  # mask areas shallower than 500m
    masked_500m = np.copy(depthm.mask) + masked  # create binary mask from depth data
    
    collection_boxes = generate_collection_boxes()
    spread_regions = generate_spreading_regions(collection_boxes, umgrid, masked, masked_500m)

    n_lat, n_lon, n_t = len(ds_discharge.latitude), len(ds_discharge.longitude), len(ds_discharge.t)
    surface_matrix_3d = np.resize(surface_matrix(ds_discharge.longitude.values, ds_discharge.latitude.values), (n_t, n_lat, n_lon))

    if 'field672' in ds_discharge.keys():
         ds_discharge.rename({'field672':'discharge'})

    def to_Sv(discharge, unit):

        # Water density
        d = 1000

        if unit == 'm3/s':
            return discharge * 10**(-6)
        elif unit == 'kg/s':
            return discharge / d * 10**(-6)
        elif unit == 'kg/m2/s':
            return discharge * surface_matrix_3d / d * 10**(-6)
        elif unit == 'Sv':
            return discharge
        else:
            raise ValueError("____ Mode not recognized")

    if units is None:
        units = ds_discharge.discharge.units
    dischargeSv = to_Sv(ds_discharge.discharge.values, units)
    
    class FluxRegion:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __repr__(self):
            return f"{self.name} -> {self.value}"

    if details=='low':
        fluxes = {'elwg': FluxRegion("East Laurentide and West Greenland", [0] * n_t), 
                'gin': FluxRegion("GIN seas", [0] * n_t),
                'med': FluxRegion("Mediterranean Sea", [0] * n_t),
                'arc': FluxRegion("Arctic", [0] * n_t),
                'so': FluxRegion("Southern Ocean", [0] * n_t),
                'pac':FluxRegion("Pacific", [0] * n_t)}
        fluxes_regions = {'elwg':['US_East_Coast', 'Gulf_of_Mexico', 'LabradorSea_BaffinBay'],
                         'gin':['Atlantic_GreenlandIceland', 'EastGreenland_Iceland', 'EastIceland', 'South_Iceland',
                          'UK_Atlantic', 'Eurasian_GINSeas'],
                         'med':['Mediterranean'],
                         'arc':['N_American_Arctic', 'Greenland_Arctic', 'Eurasian_Arctic'],
                         'so':['Antarctica_RossSea', 'Antarctica_AmundsenSea', 'Antarctica_WeddellSea',
                          'Antarctica_RiiserLarsonSea', 'Antarctica_DavisSea', 'Patagonia_Atlantic', 
                          'NorthNewZealand_Pacific', 'SouthNewZealand_Pacific'],
                         'pac':['Patagonia_Pacific', 'Russia_Pacific', 'East_Pacific']}
    elif details=='medium':
        fluxes = {'elwg': FluxRegion("East Laurentide and West Greenland", [0] * n_t), 
                   'egi': FluxRegion("East Greenland Iceland", [0] * n_t),
                   'nsbi': FluxRegion("Nordic Seas", [0] * n_t),
                   'med': FluxRegion("Mediterranean Sea", [0] * n_t),
                   'larc': FluxRegion("Laurentide Arctic", [0] * n_t),
                   'garc': FluxRegion("Greenland Arctic", [0] * n_t),
                   'earc': FluxRegion("Eurasian Arctic", [0] * n_t),
                   'so': FluxRegion("Southern Ocean", [0] * n_t),
                   'pac':FluxRegion("Pacific", [0] * n_t)}
        fluxes_regions = {'elwg':['US_East_Coast', 'Gulf_of_Mexico', 'LabradorSea_BaffinBay'],
                         'egi':['Atlantic_GreenlandIceland', 'EastGreenland_Iceland', 'EastIceland', 'South_Iceland'],
                         'nsbi':['UK_Atlantic', 'Eurasian_GINSeas'],
                         'med':['Mediterranean'],
                         'larc':['N_American_Arctic'],
                         'garc':['Greenland_Arctic'],
                         'earc':['Eurasian_Arctic'],
                         'so':['Antarctica_RossSea', 'Antarctica_AmundsenSea', 'Antarctica_WeddellSea',
                          'Antarctica_RiiserLarsonSea', 'Antarctica_DavisSea', 'Patagonia_Atlantic', 
                          'NorthNewZealand_Pacific', 'SouthNewZealand_Pacific'],
                         'pac':['Patagonia_Pacific', 'Russia_Pacific', 'East_Pacific']}
    elif details=='high':
        fluxes = {'eus': FluxRegion("US East coast", [0] * n_t), 
                   'gom': FluxRegion("Gulf of Mexico", [0] * n_t),
                   'lsbb': FluxRegion("Labrador Sea & Baffin Bay", [0] * n_t),
                   'atgi': FluxRegion("Altantic Greenland Iceland", [0] * n_t),
                   'egi': FluxRegion("East Greenland Iceland", [0] * n_t),
                   'eic': FluxRegion("East Iceland", [0] * n_t),
                   'sic': FluxRegion("South Iceland", [0] * n_t),
                   'ukat': FluxRegion("UK Atlantic", [0] * n_t),
                   'egin': FluxRegion("Eurasian GIN", [0] * n_t),
                   'med': FluxRegion("Mediterranean Sea", [0] * n_t),
                   'larc': FluxRegion("Laurentide Arctic", [0] * n_t),
                   'garc': FluxRegion("Greenland Arctic", [0] * n_t),
                   'earc': FluxRegion("Eurasian Arctic", [0] * n_t),
                   'atrs': FluxRegion("Antarctica Ross Sea", [0] * n_t),
                   'atas': FluxRegion("Antarctica Amundsen Sea", [0] * n_t),
                   'atws': FluxRegion("Antarctica Weddell Sea", [0] * n_t),
                   'atrl': FluxRegion("Antarctica Riiser Larson Sea", [0] * n_t),
                   'atds': FluxRegion("Antarctica Davis Sea", [0] * n_t),
                   'ptat': FluxRegion("Patagonia Atlantic", [0] * n_t),
                   'nnz': FluxRegion("North New Zealand", [0] * n_t),
                   'snz': FluxRegion("South New Zealand", [0] * n_t),
                   'ptpc': FluxRegion("Patagonia Pacific", [0] * n_t),
                   'rspc': FluxRegion("Russia Pacific", [0] * n_t),
                   'epc': FluxRegion("East Pacific", [0] * n_t)}
        fluxes_regions = {'eus':['US_East_Coast'],
                         'gom':['Gulf_of_Mexico'],
                         'lsbb':['LabradorSea_BaffinBay'],
                         'atgi':['Atlantic_GreenlandIceland'],
                         'egi':['EastGreenland_Iceland'],
                         'eic':['EastIceland'],
                         'sic':['South_Iceland'],
                         'ukat':['UK_Atlantic'],
                         'egin':['Eurasian_GINSeas'],
                         'med':['Mediterranean'],
                         'larc':['N_American_Arctic'],
                         'garc':['Greenland_Arctic'],
                         'earc':['Eurasian_Arctic'],
                         'atrs':['Antarctica_RossSea'],
                         'atas':['Antarctica_AmundsenSea'],
                         'atws':['Antarctica_WeddellSea'],
                         'atrl':['Antarctica_RiiserLarsonSea'],
                         'atds':['Antarctica_DavisSea'],
                         'ptat':['Patagonia_Atlantic'],
                         'nnz':['NorthNewZealand_Pacific'],
                         'snz':['SouthNewZealand_Pacific'],
                         'ptpc':['Patagonia_Pacific'],
                         'rspc':['Russia_Pacific'],
                         'epc':['East_Pacific']}
    else:
        raise ValueError("____ Option not recognized")

    for flux_region in fluxes_regions.keys():
        spread_region_loc_3d = np.zeros((n_t, n_lat, n_lon))
        for spread_region in spread_regions:
            if spread_region['name'] in fluxes_regions[flux_region]:
                spread_region_loc_3d += np.resize(spread_region['loc'].mask, (n_t, n_lat, n_lon))
        
        fluxes[flux_region].value += np.nansum(dischargeSv * spread_region_loc_3d, axis=(1, 2))
    
    fluxes['tot'] = FluxRegion("Total", np.sum([flux.value for flux in list(fluxes.values())],axis=0))
    
    # Apply running means
    if rmean:
        for key in fluxes:
            fluxes[key].values = rmean(fluxes[key].value, rmean)
    
    return fluxes

def kgm2s_to_m3s(ds_input):
    """
    Convert a discharge dataset from kg/m2/s to m3/s.
    """
    
    print("____ Converting kg/m2/s to m3/s")
    d = 1000  # water density
    ds_output = ds_input.copy().update({"discharge": np.multiply(ds_input.discharge / d, surface_matrix(ds_input.longitude.values, ds_input.latitude.values))})
    ds_output['discharge'].attrs['units'] = "m3/s"
    return  ds_output

def remove_depth(ds_input, depth_name='depth'):
    """
    Convert a discharge dataset from kg/m2/s to m3/s.
    """

    print("____ Removing depth coordinate")

    if depth_name in ds_input.coords.keys():
        return ds_input.isel({depth_name:0}).drop(depth_name)
    elif 'depth' in ds_input.coords.keys():
        return ds_input.isel({'depth':0}).drop('depth')
    elif 'unspecified' in ds_input.coords.keys():
        return ds_input.isel({'unspecified':0}).drop('unspecified')
    else:
        print("__ No depth coordinate was found")
        return ds_input

def unmask_lsm(ds_input):

    print("____ Unmasking land-sea mask")


    ds_output = ds_input.copy()
    ds_output.discharge.values = np.where(np.isnan(ds_output.discharge), 0, ds_output.discharge)
    ds_output.attrs['lsm'] = None

    return ds_output

def ancil_to_discharge(ds_input):
    """
    Convert to m3/s and remove surface depth.
    """

    print("__ Converting ancil to discharge")

    ds_output = ds_input.copy()
    ds_output = unmask_lsm(remove_depth(kgm2s_to_m3s(ds_output)))

    return ds_output

