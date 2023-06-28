import sys

from data import Lattice, Catalogue
from utils import plotting

if __name__=='__main__':
    cat_name = sys.argv[1]
    cat = Catalogue.from_file(cat_name, 0)
    
    lat_name = sys.argv[2]
    lat_data = cat[lat_name]
    lat_data.pop('edge_adjacency')
    
    lat = Lattice(**lat_data)
    lat = lat.create_windowed()
    fig = plotting.plotly_unit_cell_3d(lat, repr='fundamental', coords='transformed', show_uc_box=True)
    fig.show()
