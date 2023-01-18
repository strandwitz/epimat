import pandas as pd
from Film_Matcher_and_Imaging_Definitions import Image_Matches
import time

names = ["film id","ff","f orient","sub id","sf","s orient","fa vector",
         "fb vector","sa vector","sb vector"]
cols = [1,2,3,4,5,6,7,8,9,10]
df = pd.read_csv('C:matching_df.csv', header=0, names=names, usecols=cols)

# USER INPUT STARTS HERE
'''
Matches = the location of the match you want imaged. eg: [0], [0,2,3], etc.

Slab_half_length_x = int or float. The half length (effectively makes -val
                     to val in the x). ex. 1, 10, 12.35, etc.

Slab_half_width_y = int or float. The half width (effectively makes -val to
                    val in the y). Same examples as Slab_half_length_x.

Sub_slab_depth_z = int or float. The depth of the substrate lattice.
                   ex. 0, 1, 10, 12.35, etc. Choosing zero will give the
                   surface layer of the match only.

Silm_slab_depth_z = int or float. The depth of the film lattice.
                    Same as Sub_slab_depth_z.

Angstroms = a Bool that determines whether the above variables should be
            taken as true angstrom values or as scalar integers that will
            be multiplied by the match parameters.
            
Strain_film = a Bool that determines whether the film should be strained to
              the substrate lattice (match atoms at the same x and y value)
              True = strain the film and False = DO NOT strain the film.
              
Py_Image = a Bool that determines whether a 2D python figure should be
           generated along with the XYZ file. True = generate the python
           figure. False = DO NOT generate a python figure.
'''
matches = [2]
slab_half_length_x = 1
slab_half_width_y = 1
sub_slab_depth_z = 1
film_slab_depth_z = 1
angstroms = False
strain_film = True
py_image = False
Trigonal_Film_is_Hexagonal = True
Symmetric_20nm_Chunk = True # to use, set angstroms to False
# USER INPUT ENDS HERE

tic = time.perf_counter()
IM = Image_Matches(df, matches, py_image, slab_half_length_x,
                   slab_half_width_y, sub_slab_depth_z, film_slab_depth_z,
                   angstroms, strain_film, Trigonal_Film_is_Hexagonal,
                   Symmetric_20nm_Chunk)
IM.Get_Match_xyz_Files()
toc = time.perf_counter()
print(f"Image Matches took {toc - tic:0.1f} seconds to return generated XYZ files.")
