# utils :

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def is_in_california(polygons, lat, lng):
    offset_lat, offset_lng = 9.7, -14.3
    point = Point(lat+offset_lat, lng+offset_lng)
    for polygon in polygons:
        if polygon.contains(point):
            return True
    return False


# home:

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pickle

# ...

if "polygons" not in st.session_state:
    polygons = pickle.load(open('data/polygons.pkl', 'rb'))
    polygons = [Polygon(polygon) for polygon in polygons]
    st.session_state.polygons = polygons

# ...

if is_in_california(st.session_state.polygons, lat, lng):

