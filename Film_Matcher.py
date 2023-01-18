from pymatgen import MPRester
from Film_Matcher_and_Imaging_Definitions import FilmAnalyzer
import time
#
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
#
mpr = MPRester()

# USER INPUT STARTS HERE
'''
All material data can be observed at https://materialsproject.org/

Intro: The Materials Project API utilizes MongoDB structure for their Rester
query format. Therefore, by using key words specific materials can be chosen,
or general searches can be performed. ex. {"material_id": "mp-1265"} to select
rocksalt magnesium oxide specifically or {"material_id": {"$in": ["mp-1265",
"mp-406"]}} for multiple materials identified with their Materials Project
identifier. Also, {"elements":{"$in": ['Li','Na','Ca','Mg',"Al"], "$all":
["O"]}, "nelements":2} to get all binary oxides of the elements contained
in the list. DO NOT change the properties (the words contained in the boxy
brackets []) after the search criteria in squiggly brackets {}.

Films = sets the search parameters of the desired films.

Substrates = sets the search parameters of the desired substrates.

Film_millers = list of tuples. Miller indices of interest for the film.
               ex. [(0, 0, 1)] or [(0, 0, 1),(1,1,0),(1,1,1)] for multiple.
               [] if you have no specific millers of interest.
               
Substrate_millers = list of tuples. Miller indices of interest for the
                    substrate. Same examples as Film_millers.

Film_max_miller = integer. The maximum value within the film miller index if
                  no specific miller indices are given. Defaults to 1.

Substrate_max_miller = integer. The maximum value within the substrate miller
                       index if no specific miller indices are given.
                       Defaults to 1.
'''
#"material_id": {"$in": ["mp-661", "mp-804"]
'''
films = mpr.query({"elements": {"$all": ["O"]}, "nelements": {"$in": [2]},
                   "spacegroup.symbol": {"$nin": ["P1"]}},
                   ['structure', 'material_id',
                    'pretty_formula', 'elasticity.compliance_tensor'])
'''
'''
films = mpr.query({"material_id": {"$in": ["mp-886", "mp-1243"]}},
                   ['structure', 'material_id',
                    'pretty_formula', 'elasticity.compliance_tensor'])
'''

#print(films)

films = mpr.query({"material_id": "mp-2490"},
                  ['structure', 'material_id', 'pretty_formula',
                   'elasticity.compliance_tensor'])
'''
films = mpr.query({"material_id": {"$in": [
    "mp-2133", "mp-1143", "mp-3536", "mp-1265", "mp-5854", "mp-3427",
    "mp-2920", "mp-5304", "mp-19313", "mp-886", "mp-1243", "mp-390,",
    "mp-2657", "mp-2605", "mp-2858", "mp-2574", "mp-352", "mp-5229",
    "mp-5986", "mp-5020", "mp-2998", "mp-5777", "mp-20589", "mp-2652",
    "mp-14254", "mp-4820", "mp-4609", "mp-19845", "mp-1075921", "mp-9831",
    "mp-20337", "mp-1185357", "mp-17715", "mp-4019", "mp-216", "mp-25279",
    "mp-1102963", "mp-19399", "mp-510408", "mp-18759", "mp-19395",
    "mp-19006", "mp-19770", "mp-19306", "mp-18748", "mp-19079", "mp-19009",
    "mp-361", "mp-825", "mp-22598" ,"mp-856", "mp-2292", "mp-1968",
    "mp-20194", "mp-1045", "mp-218", "mp-1182469", "mp-504886",
    "mp-643084", "mp-2345", "mp-812", "mp-679", "mp-1767", "mp-2814",
    "mp-550893", "mp-1018721", "mp-20633", "mp-2723", "mp-20725", "mp-554867"]}},
                       ['structure', 'material_id', 'pretty_formula',
                        'elasticity.compliance_tensor'])
'''
'''


#films = [Structure.from_file("mp-1064492_CaO.cif")]

#print(films[0]['material_id)
'''
'''
substrates = mpr.query({"material_id": "mp-804"},
                       ['structure', 'material_id', 'pretty_formula'])

'''
'''
substrates = mpr.query({"material_id": {"$in": [
    "mp-5304", "mp-5229", "mp-1265", "mp-1143", "mp-804",
    "mp-2133", "mp-886", "mp-5020", "mp-6930"]}},
                       ['structure', 'material_id', 'pretty_formula'])
'''
substrates = mpr.query({"material_id": "mp-149"},
                   ['structure', 'material_id', 'pretty_formula'])



#print(films[11])

film_millers = [(1,0,0),(1,1,0),(1,1,1)]
substrate_millers = [(1,0,0),(1,1,0),(1,1,1)]
# (1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,1,1)
film_max_miller = 1
substrate_max_miller = 1
Convert_Trigonal_to_Hexagonal = True
Only_Lowest_Strain_Per_Sub_Miller = False
Use_Compliance_Tensor = True
# USER INPUT ENDS HERE

tic = time.perf_counter()
FA = FilmAnalyzer(film_max_miller, substrate_max_miller,
                  Convert_Trigonal_to_Hexagonal,
                  Only_Lowest_Strain_Per_Sub_Miller,
                  Use_Compliance_Tensor)
FA.Execute_Film_Matcher(films, substrates, film_millers, substrate_millers)
toc = time.perf_counter()
print(f"Film Matcher took {toc - tic:0.1f} seconds to return matches.")

