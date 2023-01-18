from pymatgen.core.surface import (get_symmetrically_distinct_miller_indices,
                                   get_symmetrically_equivalent_miller_indices)
from pymatgen.analysis.substrate_analyzer import (fast_norm, vec_angle,
                                                  vec_area, ZSLGenerator)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.adsorption import reorient_z, get_mi_vec
from pymatgen.core.periodic_table import Element
from pymatgen.core.surface import SlabGenerator
from pymatgen.core.structure import Structure
from pymatgen.io.xyz import XYZ
from pymatgen import MPRester
import pymatgen.core.lattice

from monty.fractions import lcm, gcd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'figure.figsize': [7,5]})
from operator import itemgetter
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import copy
import math
import os

from mpl_toolkits.mplot3d import Axes3D

class Get_Oriented_Crystal_Plane:
    def Get_LMN(self, Miller):
        '''
        Gets a scale vector for use in Get_Representative_Distances.
        The scale vector determines the minimum scalar each a, b, or c
        crystal vector needs to be multiplied so that the plane intersects
        the unit cell at whole numbers. (ex. For miller = (1, 1, 2), the
        plane represented by the miller intersects the unit cell at:
        a, b, and 1/2*c. Therefore, the scale vector = [2, 2, 1] so that
        the new intersection points are: 2*a, 2*b, and c)
        '''
        miller = list(Miller)
        GCD = gcd(miller[0],miller[1],miller[2])
        if Miller.count(0) == 2:
            LMN = []
            for m in miller:
                if m == 0:
                    LMN.append(1.0)
                else:
                    LMN.append(m/GCD)
        elif Miller.count(0) == 1:
            positions = [0, 1, 2]
            position = Miller.index(0)
            positions.remove(position)
            LCM = lcm(miller[positions[0]], miller[positions[1]])
            LMN = []
            for m in miller:
                if m == 0:
                    LMN.append(1.0)
                else:
                    LMN.append(LCM/m)
        else:
            LCM = lcm(miller[0], miller[1], miller[2])
            LMN = [LCM/miller[0], LCM/miller[1], LCM/miller[2]]
        abs_LMN = [abs(lmn) for lmn in LMN]
        return abs_LMN

    def Law_of_Cosines(self, a, b, angle_deg):
        '''
        The law of cosines for determining the length of a side of a
        non-right triangle given the other 2 side lengths and the angle
        opposite of the side.
        '''
        c = np.sqrt((a**2)+(b**2)-(2*a*b*np.cos(angle_deg*(np.pi/180))))
        return c

    def Get_Representative_Distances(self, structure, crystal_sys, SM):
        '''
        Gets the distances between the three intersection points. These
        distances represent the characteristic distances of the plane along
        a supercell of dimensions [L*a, M*b, N*c]. Gives the "edges" of the
        finite miller plane.
        '''
        LMN = self.Get_LMN(SM)
        lattice = structure.lattice
        mod_a = LMN[0]*lattice.a
        mod_b = LMN[1]*lattice.b
        mod_c = LMN[2]*lattice.c
        if SM.count(0) == 2:
            abs_miller = tuple(abs(x) for x in SM)
            if abs_miller.index(max(abs_miller)) == 0:
                rep_ab = [mod_b, mod_c]
            elif abs_miller.index(max(abs_miller)) == 1:
                rep_ab = [mod_a, mod_c]
            else:
                rep_ab = [mod_a, mod_b]
        elif SM.count(0) == 1:
            if SM.index(0) == 0:
                if min(SM) == 0 or max(SM) == 0:
                    angle_deg = lattice.alpha
                    rep_ab = [lattice.a, self.Law_of_Cosines(
                        mod_b, mod_c, angle_deg)]
                else:
                    angle_deg = 180-lattice.alpha
                    rep_ab = [lattice.a, self.Law_of_Cosines(
                        mod_b, mod_c, angle_deg)]
            elif SM.index(0) == 1:
                if min(SM) == 0 or max(SM) == 0:
                    angle_deg = lattice.beta
                    rep_ab = [self.Law_of_Cosines(
                        mod_a, mod_c, angle_deg), lattice.b]
                else:
                    angle_deg = 180-lattice.beta
                    rep_ab = [self.Law_of_Cosines(
                        mod_a, mod_c, angle_deg), lattice.b]
            else:
                if min(SM) == 0 or max(SM) == 0:
                    angle_deg = lattice.gamma
                    rep_ab = [self.Law_of_Cosines(
                        mod_a, mod_b, angle_deg), lattice.c]
                else:
                    angle_deg = 180-lattice.gamma
                    rep_ab = [self.Law_of_Cosines(
                        mod_a, mod_b, angle_deg), lattice.c] 
        else:
            LM = list(SM)
            if (LM[0] > 0 and LM[1] > 0 and LM[2] > 0) or (
                LM[0] < 0 and LM[1] < 0 and LM[2] < 0):
                angle_deg_ac = lattice.beta
                angle_deg_bc = lattice.alpha
                rep_ab = [self.Law_of_Cosines(mod_a, mod_c, angle_deg_ac),
                          self.Law_of_Cosines(mod_b, mod_c, angle_deg_bc)]
            elif (LM[0] > 0 and LM[1] < 0 and LM[2] < 0) or (
                LM[0] < 0 and LM[1] > 0 and LM[2] > 0):
                angle_deg_ac = 180-lattice.beta
                angle_deg_bc = lattice.alpha
                rep_ab = [self.Law_of_Cosines(mod_b, mod_c, angle_deg_bc),
                          self.Law_of_Cosines(mod_a, mod_c, angle_deg_ac)]
            elif (LM[0] > 0 and LM[1] < 0 and LM[2] > 0) or (
                LM[0] < 0 and LM[1] > 0 and LM[2] < 0): 
                angle_deg_bc = 180-lattice.alpha
                angle_deg_ac = lattice.beta
                rep_ab = [self.Law_of_Cosines(mod_a, mod_c, angle_deg_ac),
                          self.Law_of_Cosines(mod_b, mod_c, angle_deg_bc)]
            else:
                angle_deg_ac = 180-lattice.beta
                angle_deg_bc = 180-lattice.alpha
                rep_ab = [self.Law_of_Cosines(mod_a, mod_c, angle_deg_ac),
                          self.Law_of_Cosines(mod_b, mod_c, angle_deg_bc)]
        return rep_ab

    def Coords_Ele(self, coords, species):
        '''
        Assigns an element to a cartesian coordinate in a concise list format.
        '''
        coords_ele = []
        for num in range(0,len(coords)):
            cwe = [coords[num], species[num]]
            coords_ele.append(cwe)
        return coords_ele

    def Select_SC_Cut_Ref(self, CWE, single_lattice, crystal_sys):
        '''
        Selects a reference atom that will determine the edges of the unit
        cell after translation and cutting.
        '''
        Refs = []
        '''
        if crystal_sys == "hexagonal": ### with el select, can remove hex spec
            for coord_we in CWE:
                coord = coord_we[0]
                dx = np.abs(coord[0]-0)
                dy = np.abs(coord[1]-0)
                dz = np.abs(3*single_lattice.c-coord[2])
                xysum = dx+dy
                comp = [coord, coord_we[1], dz, xysum]
                Refs.append(comp)
        else:
            element = CWE[0][1]
            print('element',element)
            choice = []
            for cwe in CWE:
                if cwe[1] == element:
                    choice.append(cwe)
            xvs = []
            for ch in choice:
                xv = ch[0][2]
                xvs.append(xv)
            minimum = min(xvs)
            mins = []
            for point in choice:
                coord = point[0]
                
                if minimum - 0.0001 <= coord[2] <= minimum + 0.0001:
                    comp = [coord, point[1], coord[0]+coord[1], coord[1]]
                    Refs.append(comp)
        SRef = sorted(Refs, key=itemgetter(2,3))
        Ref = [SRef[0][0], SRef[0][1]]
        print('CR',Ref)
        '''
        lmat = single_lattice.matrix
        print('lmat',lmat)
        element = CWE[0][1]
        print('element',element)
        choice = []
        xvs = []
        for cwe in CWE:
            if cwe[1] == element:
                choice.append(cwe)
                xvs.append(cwe[0][2])
        '''
        xvs = []
        for ch in choice:
            xv = ch[0][2]
            xvs.append(xv)
        '''
        if lmat[2][2] < 0:
            value = max(xvs)
        else:
            value = min(xvs)
        for point in choice:    
            if value - 0.0001 <= point[0][2] <= value + 0.0001:
                comp = [point[0], point[1],
                        np.abs(point[0][0])+np.abs(point[0][1]),
                        point[0][0], point[0][1]]
                        #coord[0]+coord[1], coord[1]]
                Refs.append(comp)
        SRef = sorted(Refs, key=itemgetter(2,3,4))
        Ref = [SRef[0][0], SRef[0][1]]
        print('CR',Ref)
        return Ref
        

    def Translation_Vector_000(self, center):
        '''
        Generates the translation vector that will be used to translate
        the reference atom to [0, 0, 0].
        '''
        TV = [0-center[0], 0-center[1], 0-center[2]]
        return TV

    def Translate_Ref_to_000(self, points, Ref):
        '''
        Translates a group of points such that a reference point is located
        at [0, 0, 0].
        '''
        Translation = []
        ref = Ref[0]
        TV = self.Translation_Vector_000(ref)
        for pointWE in points:
            point = pointWE[0]
            a = point[0]+TV[0]
            b = point[1]+TV[1]
            c = point[2]+TV[2]
            TP = np.array([a,b,c])
            TWE = [TP, pointWE[1]]
            Translation.append(TWE)
        return Translation

    def Cut_Supercell(self, points, lattice, repeat, crystal_sys): #error here
        '''
        Takes the generated supercell and cuts it such that there is an atom
        on every corner of the unit cell.
        '''
        lmat = lattice.matrix
        scaled_lmat = []
        for pt in lmat:
            spt = np.array([x*(repeat-1) for x in pt])
            scaled_lmat.append(spt)
        print('scaled lmat',scaled_lmat)
        '''
        if crystal_sys == "hexagonal": ##
            UC_Bounds = [np.array([0,0,0]), scaled_lmat[0], scaled_lmat[1],
                         -1*scaled_lmat[2]]
        else:
        '''
        UC_Bounds = [np.array([0,0,0]), scaled_lmat[0],
                     scaled_lmat[1], scaled_lmat[2]]
        u = np.cross(UC_Bounds[0]-UC_Bounds[1],UC_Bounds[0]-UC_Bounds[2])
        v = np.cross(UC_Bounds[0]-UC_Bounds[1],UC_Bounds[0]-UC_Bounds[3])
        w = np.cross(UC_Bounds[0]-UC_Bounds[2],UC_Bounds[0]-UC_Bounds[3])
        up0_up3 = [np.dot(u,UC_Bounds[0]), np.dot(u,UC_Bounds[3])]
        vp0_vp2 = [np.dot(v,UC_Bounds[0]), np.dot(v,UC_Bounds[2])]
        wp0_wp1 = [np.dot(w,UC_Bounds[0]), np.dot(w,UC_Bounds[1])]
        print('u v w', u, v, w)
        print('dots',up0_up3,vp0_vp2,wp0_wp1)
    
        Cut_UC = []
        for pointWE in points:
            point = np.array(pointWE[0])
            ux = np.dot(u,point)
            vx = np.dot(v,point)
            wx = np.dot(w,point)
            #print('ux vx wx',ux,vx,wx)
            if (min(up0_up3)-0.1 <= ux <= max(up0_up3)+0.1 and
                min(vp0_vp2)-0.1 <= vx <= max(vp0_vp2)+0.1 and
                min(wp0_wp1)-0.1 <= wx <= max(wp0_wp1)+0.1):
                Cut_UC.append(pointWE)
                #print('pass')
            else:
                pass
        return Cut_UC

    def Cut_SC_Trans_Ref(self, CWE, lattice, repeat):
        '''
        Gets the atom which will occupy the [0, 0, 0] position for hexagonal
        or trigonal lattices. The cut SC reference is on the top of the SC
        for the hexagonal lattices whereas the cut SC ref for the other
        bravais lattices are on the bottom. For the hexagonal lattices to
        have a SC with whose atoms have z >= 0, a reference on the bottom
        of the SC must be chosen and moved to the [0, 0, 0] position.
        '''
        Refs = []
        for point in CWE:
            comp = [point[0], point[1],
                    point[0][2], point[0][0], point[0][1]] # z, x, y
            Refs.append(comp)
        # SRef = sorted(Refs, key=itemgetter(2,3,4)) OLD
        Sref = Refs.sort(key=lambda k: (k[2],  -k[3], -k[4]), reverse=True)
        Ref = [SRef[0][0], SRef[0][1]]
        print('TR',Ref)
        return Ref

    def Adjust_Lmat(self, LMat, tref): # not used please delete
        print('tref',tref)
        pot_pts = [-1, 0, 1]
        solution = []
        for l in pot_pts:
            for m in pot_pts:
                for n in pot_pts:
                    add = (np.array(tref)+np.array(l*LMat[0])+
                             np.array(m*LMat[1])+np.array(n*LMat[2]))
                    check = [-0.01 <= ADD <= 0.01 for ADD in add]
                    #print('all check', all(check))
                    if all(check):
                        LMN = [l,m,n]
                        almn = []
                        for lmn in LMN:
                            if lmn == 0:
                                almn.append(1)
                            else:
                                almn.append(lmn)
                        return almn

    def Select_Plane_Reference(self, CWE, lattice, miller, repeat):
        '''
        Selects an atom on one of the corners of the unit cell that is
        known to be located on the plane defined by the given miller index.
        A crystal can contain many planes that represent the given miller.
        Therefore, choosing a reference locks the plane location within the
        supercell.
        '''
        LMAT = lattice.matrix
        adj_lmat = [LMA*(repeat-1) for LMA in LMAT]
        '''
        print('OG lmat',LMat)
        solution = self.Adjust_Lmat(LMat, tref[0])
        adj_lmat = [solution[0]*LMat[0], solution[1]*LMat[1],
                    solution[2]*LMat[2]]
        print('adj LMAT', adj_lmat)
        '''
        refs = [adj_lmat[2], adj_lmat[2]+adj_lmat[1],
                adj_lmat[2]+adj_lmat[0], [0,0,0]]
        
        LM = list(miller)
        if ((LM[0] > 0 and LM[1] > 0 and LM[2] < 0) or (
            LM[0] < 0 and LM[1] > 0 and LM[2] > 0)) or (
                (min(miller) < 0 and max(miller) > 0 and LM[1] == 0)):
            for pointWE in CWE:
                point = pointWE[0]
                if np.around(point[0],3) == np.around(refs[3][0],3) and np.around(
                    point[1],3) == np.around(refs[3][1],3) and np.around(
                        point[2],3) == np.around(refs[3][2],3):
                    return pointWE
        elif (LM[0] > 0 and LM[1] < 0 and LM[2] < 0) or (
            LM[0] < 0 and LM[1] > 0 and LM[2] > 0):
            for pointWE in CWE:
                point = pointWE[0]
                if np.around(point[0],3) == np.around(refs[2][0],3) and np.around(
                    point[1],3) == np.around(refs[2][1],3) and np.around(
                        point[2],3) == np.around(refs[2][2],3):
                    return pointWE
        elif ((min(miller) < 0 and max(miller) > 0) and LM[0] == 0) or (
            (LM[0] > 0 and LM[1] > 0 and LM[2] == 0) or (
                LM[0] < 0 and LM[1] < 0 and LM[2] == 0)) or (
                    (LM[0] > 0 and LM[1] < 0 and LM[2] > 0) or (
                        LM[0] < 0 and LM[1] > 0 and LM[2] < 0)):
            for pointWE in CWE:
                point = pointWE[0]
                if np.around(point[0],3) == np.around(refs[1][0],3) and np.around(
                    point[1],3) == np.around(refs[1][1],3) and np.around(
                        point[2],3) == np.around(refs[1][2],3):
                    return pointWE
        else:
            for pointWE in CWE:
                point = pointWE[0]
                if np.around(point[0],3) == np.around(refs[0][0],3) and np.around(
                    point[1],3) == np.around(refs[0][1],3) and np.around(
                        point[2],3) == np.around(refs[0][2],3):
                    return pointWE

    def Compare(self, point1, point2, point3):
        '''
        Checks that three points are not equal to each other.
        '''
        rp1 = str([np.around(x,3) for x in point1])
        rp2 = str([np.around(y,3) for y in point2])
        rp3 = str([np.around(z,3) for z in point3])
        if (rp3 != rp1 and rp3 != rp2 and rp1 != rp2):
            return True
        return False

    def Miller_Trunc_Comp(self, index, MIM):
        '''
        Truncates the rounded miller from Pymatgen and compares it to the
        desired miller within a tolerance.
        '''
        #trunc_p = [float(str(float(x))[:-1]) for x in index]
        count = 0
        for miller in MIM:
            if ((miller[0] - 0.001 < index[0] < miller[0] + 0.001) and (
                miller[1] - 0.001 < index[1] < miller[1] + 0.001) and (
                    miller[2] - 0.001 < index[2] < miller[2] + 0.001)):
                count = count+1
        if count > 0:
            return True
        else:
            return False

    def Vector_Sub(self, p1, p2):
        '''
        Takes two points in 3D and gets the distance vector from p1 to p2.
        '''
        vec = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
        return vec

    def Get_Init_Refs(self, reordered_ele, Ref, lattice, MIM):
        '''
        Using the corner reference as one point, cycles through all the
        points in the SC to find the first three points on the specific
        miller plane.
        '''
        p1 = Ref[0]
        print('mim', MIM)
        for point2 in reordered_ele:
            p2 = point2[0]
            for point3 in reordered_ele:
                p3 = point3[0]
                comp1 = self.Compare(p1, p2, p3)
                p1p2 = self.Vector_Sub(p1, p2)
                p1p3 = self.Vector_Sub(p1, p3)
                comp2 = np.around(vec_area(p1p2, p1p3),3)
                if ((p2[0] != 0 or p2[1] != 0) and (
                    p3[0] != 0 or p3[1] != 0) and (
                        comp1 == True and comp2 != 0)):
                    comp_coords = [p1, p2, p3]
                    index = lattice.get_miller_index_from_coords(comp_coords)
                    if (24.8 < p2[1] < 25 and -0.1 < p2[0] < 0.1 and
                        6 < p3[0] < 6.2 and 11.3 < p3[2] < 11.5):
                        print('p2, p3, index',p2,p3,index)
                    COMP = self.Miller_Trunc_Comp(index, MIM)
                    if COMP == True:
                        return [Ref, point2, point3]

    def Get_All_Unique_Pts_on_Plane(self, reordered_ele, Ref, structure, miller):
        '''
        Uses a reference and two other initial points to determine all other
        unique points that reside on the given miller plane.
        '''
        MIM = [miller, tuple([-1*x for x in miller])]
        lattice = structure.lattice
        Init_Refs = self.Get_Init_Refs(reordered_ele, Ref, lattice, MIM)
        print('IR',Init_Refs)
        
        AUPoPwE = []
        IR_str = []
        for IR in Init_Refs:
            AUPoPwE.append(IR)
            IR_str.append(str(IR[0]))
        p1 = Init_Refs[0][0]
        for point2 in reordered_ele:
            p2 = point2[0]
            p1p2 = self.Vector_Sub(p1, p2)
            p1p3a = self.Vector_Sub(p1, Init_Refs[1][0])
            p1p3b = self.Vector_Sub(p1, Init_Refs[2][0])
            comp1 = np.around(vec_area(p1p2, p1p3a),3)
            comp2 = np.around(vec_area(p1p2, p1p3b),3)
            if ((comp1 != 0 or comp2 != 0) and str(p2) not in IR_str):
                if comp1 != 0:
                    point3 = Init_Refs[1]
                else:
                    point3 = Init_Refs[2]
                p3 = point3[0]
                comp_coords = [p1, p2, p3]
                index = lattice.get_miller_index_from_coords(comp_coords)
                #RIndex = tuple(np.around(ind,3) for ind in index)
                #print('comp pt',p2,index)
                COMP = self.Miller_Trunc_Comp(index, MIM)
                if COMP == True:
                    AUPoPwE.append(point2)
                else:
                    pass
            else:
                pass
        #print('POP we', AUPoPwE, len(AUPoPwE))
        return AUPoPwE

    def Pick_Rep_Atoms(self,  PTS, RAt, miller):
        '''
        Using the characteristic distances along the crystal face, selects
        the atoms that represent those distances from the atoms that reside
        on the given miller plane.
        '''
        points = [] # NEW
        for atom in PTS: # NEW
            if np.around(atom[0][0],2) >= 0 and np.around(atom[0][1],2) >= 0: # NEW
                points.append(atom) # NEW
        #print('pts',points)
        if RAt[0] == RAt[1]:
            reps = []
            for PWE in points:
                point = PWE[0]
                distance = np.around(
                    np.sqrt(point[0]**2+point[1]**2+point[2]**2),3)
                if distance == np.around(RAt[0],3): #and np.around(
                    #point[1],3) >= 0:
                    reps.append(PWE)
                    if len(reps) == 2:
                        break
            #print("reps",reps,reps[0][0],reps[0][0][0])
            '''
            if miller.count(0) == 2:
                mm = max(miller)
                if miller.index(mm) == 0:
                    if np.abs(reps[0][0][1]) > np.abs(reps[1][0][1]):
                        rep_atoms = reps
                    else:
                        rep_atoms = reps[::-1]
                elif miller.index(mm) == 1:
                    if np.abs(reps[0][0][0]) > np.abs(reps[1][0][0]):
                        rep_atoms = reps
                    else:
                        rep_atoms = reps[::-1]
                else:
                    if np.abs(reps[0][0][1]) < np.abs(reps[1][0][1]):
                        rep_atoms = reps
                    else:
                        rep_atoms = reps[::-1]
            else:
                if reps[0][0][1] < reps[1][0][1]:
                    rep_atoms = reps
                else:
                    rep_atoms = reps[::-1]
        '''
        else:
            reps = [0,0]
            for PWE in points:
                point = PWE[0]
                distance = np.around(np.sqrt(point[0]**2+point[1]**2+point[2]**2),3)
                print('p d',point,distance)
                if distance == np.around(RAt[0],3):
                    reps[0] = PWE
                elif distance == np.around(RAt[1],3):
                    reps[1] = PWE
        if np.abs(reps[0][0][0]) >= np.abs(reps[1][0][0]):
            rep_atoms = reps
        else:
            rep_atoms = [reps[1], reps[0]] # NEW
        #print('RAt',RAt)
        #print('RA',rep_atoms)
        return rep_atoms

    def Rotate_Around_Z(self, vector, theta):
        '''
        Rotates a vector around the z-axis.
        '''
        cos = np.cos(theta)
        sin = np.sin(theta)
        Rz = np.array(((cos, -sin, 0), (sin, cos, 0), (0, 0, 1)))
        rotationZ = np.dot(Rz, vector)
        return rotationZ

    def Rotate_Around_Y(self, vector, theta):
        '''
        Rotates a vector around the y-axis.
        '''
        cos = np.cos(theta)
        sin = np.sin(theta)
        Ry = np.array(((cos, 0, sin), (0, 1, 0), (-sin, 0, cos)))
        rotationY = np.dot(Ry, vector)
        return rotationY

    def Rotate_Around_X(self, vector, theta):
        '''
        Rotates a vector around the x-axis.
        '''
        cos = np.cos(theta)
        sin = np.sin(theta)
        Rx = np.array(((1, 0, 0), (0, cos, -sin), (0, sin, cos)))
        rotationX = np.dot(Rx, vector)
        return rotationX

    def Rotate(self, vectors, theta, s, r, el_in, el_out):
        '''
        Rotates a set of points around a specified axis.
        '''
        aligned_vecs = []
        for vectorWE in vectors:
            if el_in == True:
                vector = vectorWE[0]
            else:
                vector = vectorWE
            reshape3x1 = vector.reshape(3,1)
            if s == 0:
                R = self.Rotate_Around_Z(reshape3x1, theta)
            elif s == 1:
                R = self.Rotate_Around_Y(reshape3x1, theta)
            else:
                R = self.Rotate_Around_X(reshape3x1, theta)
            if r == True:
                Align = np.array([np.around(R[0][0],8), np.around(
                    R[1][0],8), np.around(R[2][0],8)])
            else:
                Align = np.array([R[0][0], R[1][0], R[2][0]])
            if el_out == True:
                AlignWE = [Align, vectorWE[1]]
                aligned_vecs.append(AlignWE)
            else:
                aligned_vecs.append(Align)
        return aligned_vecs

    def Reorient_Plane(self, points, RA, el_in, el_out):
        '''
        Determines the angle between a given reference vector and a chosen
        axis. Then rotates a set of points around an axis by the angle
        determined using the reference. This is done for the x, y, and z axes
        such that all the points have a z = 0.
        '''
        #print('RA',RA)
        if el_in == True and el_out == True:
            ELIZ = True
            ELOZ = True
            ELIXY = True
            ELOXY = True
        elif el_in == False and el_out == False:
            ELIZ = False
            ELOZ = False
            ELIXY = False
            ELOXY = False
        else:
            ELIZ = True
            ELOZ = False
            ELIXY = False
            ELOXY = False
        ELIRA = True
        ELORA = True
        
        XY_MAG = np.sqrt((RA[0][0][0]**2)+(RA[0][0][1]**2))
        if RA[0][0][0] < 0:
            RAZA = (-1*np.pi)-(-1*np.arcsin(RA[0][0][1]/XY_MAG))
        else:
            RAZA = -1*np.arcsin(RA[0][0][1]/XY_MAG)
        SZ = 0
        RZ = False
        Align_to_X_Axis = self.Rotate(points, RAZA, SZ, RZ, ELIZ, ELOZ)
        AXARA = self.Rotate(RA, RAZA, SZ, RZ, ELIRA, ELORA)
        
        XZ_MAG = np.sqrt((AXARA[0][0][0]**2)+(AXARA[0][0][2]**2))
        RAYA = np.arcsin(AXARA[0][0][2]/XZ_MAG)
        SY = 1
        RY = False
        Rotate_Around_Y = self.Rotate(Align_to_X_Axis, RAYA,
                                      SY, RY, ELIXY, ELOXY)
        RAYRA = self.Rotate(AXARA, RAYA, SY, RY, ELIRA, ELORA)
        
        YZ_MAG = np.sqrt((RAYRA[1][0][1]**2)+(RAYRA[1][0][2]**2))
        if RAYRA[1][0][1] < 0:
            RAXA = np.pi-(-1*np.arcsin(RAYRA[1][0][2]/YZ_MAG))
        else:
            RAXA = -1*np.arcsin(RAYRA[1][0][2]/YZ_MAG)
        SX = 2
        RX = True
        Rotate_Around_X = self.Rotate(Rotate_Around_Y, RAXA,
                                      SX, RX, ELIXY, ELOXY)
        RAXRA = self.Rotate(RAYRA, RAXA, SX, RX, ELIRA, ELORA)
        return Rotate_Around_X, RAXRA, RAZA, RAYA, RAXA

    def Make_Extended_Lattice(self, points, RRA):
        '''
        Makes an extended lattice in the x and y dimensions. The periodicity
        of the crystal face is preserved by using the rotated characteristic
        reference atoms to translate the atoms on the face within a given area.
        '''
        angle = Vec_Angle_Rad(RRA[0][0], RRA[1][0])
        distance = np.around(np.sqrt(
            RRA[1][0][0]**2+RRA[1][0][1]**2+RRA[1][0][2]**2),3)
        y_only = np.sin(angle)*distance
        
        y_min_max = [-4*y_only, 4*y_only]
        x_min_max = [-4*RRA[0][0][0], 4*RRA[0][0][0]]
        #y_min_max = [-18,18]
        #x_min_max = [-25, 25]
        EL = []
        tuples = []
        for n in range(-8,8): ### should be 8
            y_trans = n*RRA[1][0]
            for m in range(-8,8): ### should be 8
                x_trans = m*RRA[0][0]
                Copy = copy.deepcopy(points)
                tcopy = []
                for p in Copy:
                    NP = p+y_trans+x_trans
                    tcopy.append(NP)
                for point in tcopy:
                    rpoint = [np.around(x,8) for x in point]
                    rtup = [np.around(x,3) for x in point]
                    if ((y_min_max[0] - .01) <= rtup[1] <= (
                        y_min_max[1] + .01) and (
                            x_min_max[0] - .01) <= rtup[0] <= (
                                x_min_max[1] + .01)):
                        if tuple(rtup) in tuples:
                            pass
                        else:
                            tuples.append(tuple(rtup))
                            EL.append(rpoint)
                    else:
                        pass
        return EL

    def Symmetra(self, Copy):
        '''
        Exerts the symmetry requirements on the extended lattice and stores
        the positions of atoms that fail the symmetry requirement and need
        to be removed in a list. Any point that does not have an atom at its
        inverse does not contribute to the 2D unit cell (is a basis of the 2D
        unit cell).
        '''
        delete = []
        for n in range(0, len(Copy)):
            RIp = [-1*np.around(x,3) for x in Copy[n]]
            count = 1
            for Cp in Copy:
                RCp = [np.around(y,3) for y in Cp]
                if RIp[0] == RCp[0] and RIp[1] == RCp[1] and RIp[2] == RCp[2]:
                    break
                else:
                    if count == len(Copy):
                        delete.append(n)
                    else:
                        count = count+1
        return delete

    def Remove_Extra(self, symm, points):
        '''
        Removes the points that fail the symmetry requirement in Symmetra
        '''
        if symm != []:
            remove = points
            toremove = symm
            for m in sorted(toremove, reverse=True):
                del remove[m]
        return points

    def Get_Reduced_UC(self, points):
        '''
        Takes the atoms that make up the 2D unit cell and chooses the first
        and second nearest neighbor atoms that are not collinear and have an
        angle between them less than 90 deg. Basically chooses the primitive
        vector set of the 2D unit cell defined by the atoms on the plane.
        '''
        PWL = []
        for pt in points:
            if np.around(pt[0],3) >= 0:
                distance = np.around(np.sqrt(pt[0]**2+pt[1]**2+pt[2]**2),3)
                if np.around(pt[0],3) == 0 and np.around(
                    pt[1],3) == 0 and np.around(pt[2],3) == 0:
                    angle = 0.0
                else:
                    unit_v = [p / np.linalg.norm(pt) for p in pt]
                    angle = np.around(vec_angle(np.array(
                        [0,-1,0]), unit_v)*(180/np.pi),3)
                pwl = [pt, distance, angle]
                PWL.append(pwl)
        ordered = sorted(PWL, key=itemgetter(1,2))
        #print(ordered)
        TNT = [2,3] ###
        #TC = 0
        for num in range(1,len(ordered)):
            TC = 0
            p1t = ordered[num][0]
            print(plt)
            for n in TNT:
                for check in ordered:
                    cp = check[0]
                    if (np.around(n*p1t[0],3) == np.around(cp[0],3)) and (
                        np.around(n*p1t[1],3) == np.around(cp[1],3)) and (
                            np.around(n*p1t[2],3) == np.around(cp[2],3)):
                        TC = TC+1
                        break
            if TC == 2:
                p1 = p1t
                rem_nums = num+1
                break
        #print('p1',p1)
        RUC = []
        RUC.append(np.array(p1))
        for num2 in range(rem_nums,len(ordered)):
            TC2 = 0
            p2t = ordered[num2][0]
            angle = np.around(Vec_Angle_Rad(RUC[0], p2t)*(180/np.pi),4)
            #print('RUC0',RUC[0])
            ### changed from ordered[1][0]
            det = np.around((p1[0]*p2t[1])-(p1[1]*p2t[0]),4)
            #print('p2t ang det',p2t,angle,det)
            if angle <= 90.0 and det!= 0.0:
                for n in TNT:
                    for check2 in ordered:
                        cp2 = check2[0]
                        if (np.around(n*p2t[0],3) == np.around(cp2[0],3)) and (
                        np.around(n*p2t[1],3) == np.around(cp2[1],3)) and (
                            np.around(n*p2t[2],3) == np.around(cp2[2],3)):
                            TC2 = TC2+1
                            break
                if TC2 == 2:
                    p2 = p2t
                    RUC.append(np.array(p2))
                    #print('p2',p2)
                    return RUC

    def Get_Red_UC(self, RP, Rot_RA): #search
        print(Rot_RA)
        '''
        for test in RP:
            x = test[0]
            y = test[1]
            plt.scatter(x,y)
        plt.show()
        '''
        MEL = self.Make_Extended_Lattice(RP, Rot_RA) # OLD
        '''
        for test in MEL:
            x = test[0]
            y = test[1]
            plt.scatter(x,y)
        plt.show()
        '''
        Symm = self.Symmetra(MEL) # OLD
        #Symm = self.Symmetra(RP) # NEW
        REx = self.Remove_Extra(Symm, MEL) # OLD
        #REx = self.Remove_Extra(Symm, RP) # NEW
        Red_UC = self.Get_Reduced_UC(REx)
        print('RUC',Red_UC)
        '''
        for test in REx:
            x = test[0]
            y = test[1]
            plt.scatter(x,y)
        plt.show()
        '''
        return Red_UC

    def Get_D_Spacing(self, miller, lattice, CS): # fixed
        '''
        Calculates the d-spacing (spacing between equivalent miller planes)
        for a crystal structure given the miller index, lattice parameters,
        and crystal system. #REFERENCE HERE#
        '''
        [a, b, c] = [lattice.a, lattice.b, lattice.c]
        [alpha, beta, gamma] = [math.radians(lattice.alpha), math.radians(
            lattice.beta), math.radians(lattice.gamma)]
        
        V = a*b*c*np.sqrt(1-(np.cos(alpha)**2)-(np.cos(beta)**2)-(
                np.cos(gamma)**2)+(2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)))
        
        S11 = (b**2)*(c**2)*(np.sin(alpha)**2)
        S22 = (a**2)*(c**2)*(np.sin(beta)**2)
        S33 = (a**2)*(b**2)*(np.sin(gamma)**2)
        S12 = a*b*(c**2)*((np.cos(alpha)*(np.cos(beta)))-np.cos(gamma))
        S23 = c*b*(a**2)*((np.cos(beta)*(np.cos(gamma)))-np.cos(alpha))
        S13 = a*c*(b**2)*((np.cos(gamma)*(np.cos(alpha)))-np.cos(beta))
        d = V/np.sqrt((S11*(miller[0]**2))+(S22*(miller[1]**2))+(
            S33*(miller[2]**2))+(2*S12*miller[0]*miller[1])+(
                2*S23*miller[1]*miller[2])+(2*S13*miller[0]*miller[2]))
        return np.around(d,8)

    def Get_Z_Trans_Length(self, lattice, miller, DS): ###
        '''
        Takes the miller and the lattice matrix and gives the distance to the
        next atomically equivalent miller plane to the simplified user given
        or symmetrically equivalent miller plane. Also checks that it is some
        integer multiple of the d_spacing before returning the value.
        '''
        '''
        print("RA",RA)
        RA_cross = np.cross(RA[0][0],RA[1][0])
        print("RA Cross",RA_cross)
        RA_UV = [np.around(p/fast_norm(RA_cross),8) for p in RA_cross]
        print("RA UV",RA_UV)
        lmat = lattice.matrix
        #print('labc',lattice.a, lattice.b, lattice.c)
        print('lmat',lmat)
        adj_lmat = []
        for num in range(0,len(miller)):
            new_vec = np.array([n*miller[num] for n in lmat[num]])
            adj_lmat.append(new_vec)
        MV = adj_lmat[0]+adj_lmat[1]+adj_lmat[2]
        print('MV',MV)
        MMV = fast_norm(MV)
        print('mmv',MMV)
        MV_UV = [np.around(p2/fast_norm(MV),8) for p2 in MV]
        print("MV UV",MV_UV)
        
        miller_cart_offset_angle = Vec_Angle_Rad(RA_UV, MV_UV)
        print('MCOA',miller_cart_offset_angle*(180/np.pi))
        z_trans = np.abs(MMV*np.cos(miller_cart_offset_angle))
        '''
        z_trans = np.sqrt((lattice.a*miller[0])**2+(
            lattice.b*miller[1])**2+(lattice.c*miller[2])**2)
        # Above only works for all angles 90, need to use law of cosines version
        
        print('ZT DS',z_trans, DS)
        check = np.around(z_trans/DS,5)
        print('check please',check,check-np.around(check,0))
        if check-np.around(check,0) == 0:
            #print('is ok')
            return z_trans
        else:
            print('GET REKT')

    def Rem_Max(self, FA, pos):
        Rem_Max = max([sublist1[0][pos] for sublist1 in FA])
        x_i = [i for i, x in enumerate(
            [sublist1[0][pos] for sublist1 in FA]) if Rem_Max-0.01 <= x <= Rem_Max+0.01]
        #print(x_i)
        remains = self.Remove_Extra(x_i,FA)
        return remains

    def Get_Slab_Face_with_RE(self, slabs, repeat_miller, RE): # NEW
        '''
        Gets atoms on the face of the slab that contains the representative
        element. Cation for the substrate and anion for the film.
        '''
        Pass = False
        for pos, slab in enumerate(slabs):
            #print("original slab matrix",slab._lattice.matrix)
            slab.make_supercell([repeat_miller,repeat_miller,1])
            OES = reorient_z(slab)
            #print("OES slab matrix",OES._lattice.matrix)

            sc_species = OES.species # NEW
            init_coords = OES.cart_coords # NEW
        
            coords_ele = self.Coords_Ele(init_coords, sc_species)
            z_max = np.around(
                max([sublist[0][2] for sublist in coords_ele]),2)
            face_atoms = []
            for CE in coords_ele:
                if np.around(CE[0][2],2) == z_max:
                    face_atoms.append(CE)
                    if str(CE[1]) == RE:
                        Pass = True
            if Pass == True:
                break
        
        if Pass == False:
            pos = 0
            IS = slabs[0]
            IS.make_supercell([repeat_miller,repeat_miller,1])
            OES2 = reorient_z(IS)

            sc_species2 = OES2.species # NEW
            init_coords2 = OES2.cart_coords # NEW
            coords_ele2 = self.Coords_Ele(init_coords2, sc_species2)
            
            ZREMs = self.Rem_Max(coords_ele2, 2)
            z_max2 = np.around(
                max([sublist2[0][2] for sublist2 in ZREMs]),2)
            Pass2 = False
            while True:
                face_atoms = []
                for CE2 in ZREMs:
                    #print(z_max2,np.around(CE2[0][2],2),CE2[1],RE)
                    if np.around(CE2[0][2],2) == z_max2:
                        face_atoms.append(CE2)
                        if str(CE2[1]) == RE:
                            #print("pass is true in pass false")
                            Pass2 = True
                if Pass2 == True:
                    print("break init")
                    break
                ZREMs = self.Rem_Max(face_atoms, 2)
        #print(slab.lattice)
        #print(len(face_atoms))
        return face_atoms, pos

    def Get_Middle_Of_Face_Atoms(self, FaceAt, RE):
        '''
        print(len(FaceAt))
        fig = plt.figure()
        ax = fig.add_subplot(111)#, projection='3d')
        for point in FaceAt:
            x = np.around(point[0][0],3)
            y = np.around(point[0][1],3)
            ax.scatter(x,y, c='k')
        plt.show()
        '''
        '''
        XLS = set([np.around(sublist1[0][0],2) for sublist1 in FaceAt])
        XUL = list(XLS)
        YLS = set([np.around(sublist3[0][1],2) for sublist3 in FaceAt])
        YUL = list(YLS)
        
        if len(XUL)%2 == 0 and len(YUL)%2 == 0:
            XREMs = self.Rem_Max(FaceAt, 0)
            REMs = self.Rem_Max(XREMs, 1)
        elif len(XUL)%2 == 0:
            REMs = self.Rem_Max(FaceAt, 0)
        elif len(YUL)%2 == 0:
            REMs = self.Rem_Max(FaceAt, 1)
        else:
            REMs = FaceAt

        fig = plt.figure()
        ax = fig.add_subplot(111)#, projection='3d')
        for point in REMs:
            x = np.around(point[0][0],3)
            y = np.around(point[0][1],3)
            ax.scatter(x,y, c='k')
        plt.show()
       
        x_max = max([sublist1[0][0] for sublist1 in REMs])
        x_min = min([sublist2[0][0] for sublist2 in REMs])
        y_max = max([sublist3[0][1] for sublist3 in REMs])
        y_min = min([sublist4[0][1] for sublist4 in REMs])
        mid_vals = [(x_max+x_min)/2, (y_max+y_min)/2]
        #print(FaceAt)
        print(x_max,x_min,y_max,y_min,'mid vals',mid_vals)
        #ref_diffs = [x_max, y_max]
        for At in FaceAt:
            if str(At[1]) == RE:
                c_diffs = [np.abs(mid_vals[0]-At[0][0]),
                           np.abs(mid_vals[1]-At[0][1])]
                if -0.01 <= c_diffs[0] <= 0.01 and -0.01 <= c_diffs[1] <= 0.01:
                    T_Ref = At
                    break
        '''
        FaceAtC = copy.deepcopy(FaceAt)
        x_max = max([sublist1[0][0] for sublist1 in FaceAtC])
        x_min = min([sublist1[0][0] for sublist1 in FaceAtC])
        while True:
            x_mid = ((x_max-x_min)/2)+x_min
            #print("XM",x_mid)
            Xmids = [sublist1 for sublist1 in FaceAtC if np.around(
                sublist1[0][0],2) == np.around(x_mid,2)]
            if len(Xmids) == 0:
                FaceAtC = self.Rem_Max(FaceAtC, 0)
                x_max = max([sublist1[0][0] for sublist1 in FaceAtC])
            else:
                break
        print(Xmids)
        y_max = max([sublist2[0][1] for sublist2 in Xmids])
        y_min = min([sublist2[0][1] for sublist2 in Xmids])
        while True:
            y_mid = np.around((y_max-y_min)/2+y_min,2)
            #print("YM",y_mid)
            SL = [np.around(sublist2[0][1],2) for sublist2 in Xmids]
            if y_mid in SL:
                mid_pos = SL.index(y_mid)
                T_Ref = Xmids[mid_pos]
                break
            else:
                Xmids = self.Rem_Max(Xmids, 1)
                y_max = max([sublist2[0][1] for sublist2 in Xmids])
        return T_Ref
        
    def GOCP(self, structure, crystal_sys, miller, RE, ELI=True, ELO=False):
        '''
        Runs the process for getting a crystal face defined by a miller and
        oriented such that z = 0 for all points.
        '''
        max_miller = max(np.abs(miller)) ###
        repeat = (2*max_miller)+1
        lattice = structure.lattice
        print('lmat',lattice.matrix)
        d_spacing = self.Get_D_Spacing(miller, lattice, crystal_sys)
        ZT = self.Get_Z_Trans_Length(lattice, miller, d_spacing)
        RAt = self.Get_Representative_Distances(structure, crystal_sys, miller)
        #print('RAt',RAt)
        #sc_structure = structure * (repeat, repeat, repeat) OLD
        #sc_lattice = sc_structure.lattice OLD
        #sc_species = sc_structure.species OLD
        #init_coords = sc_structure.cart_coords OLD
        slabgen = SlabGenerator(structure, miller, repeat, 3,
                                lll_reduce=True, primitive=False,
                                in_unit_planes=True)
        slabs = slabgen.get_slabs()
        Optimum_Slab_Face_Atoms_WE, WastePos = self.Get_Slab_Face_with_RE(
            slabs, repeat, RE)
        
        MRef = self.Get_Middle_Of_Face_Atoms(Optimum_Slab_Face_Atoms_WE, RE)
        Translate = self.Translate_Ref_to_000(Optimum_Slab_Face_Atoms_WE, MRef)
        TR_NoE = [TRE[0] for TRE in Translate]
        #'''
        fig = plt.figure()
        ax = fig.add_subplot(111)#, projection='3d')
        for point in Translate:
            x = np.around(point[0][0],3)
            y = np.around(point[0][1],3)
            ax.scatter(x,y, c='k')
        plt.show()
        #'''
        Rep_Atoms = self.Pick_Rep_Atoms(Translate, RAt, miller)
        print(Rep_Atoms)
        RA_NoE = [[REE[0]] for REE in Rep_Atoms]
        print(RA_NoE)
        
        #coords_ele = self.Coords_Ele(init_coords, sc_species) OLD placement
        
        #CRef = self.Select_SC_Cut_Ref(coords_ele, lattice, crystal_sys) OLD
        #CRef = self.Cut_SC_Trans_Ref(coords_ele, lattice, crystal_sys) OLD
        #Translate = self.Translate_Ref_to_000(coords_ele, CRef)
        #face_atoms = [] NEW
        #for atoms in Translate: NEW
            #if atoms[0][2] == 0: NEW
                #face_atoms.append(atoms) NEW
        #Rep_Atoms = self.Pick_Rep_Atoms(face_atoms, RAt, miller)
        '''###
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for point in Translate:
            test = point[0]
            x = np.around(test[0],3)
            y = np.around(test[1],3)
            z = np.around(test[2],3)
            ax.scatter(x,y,z, c='k')
        plt.show()
        '''### Trans is correct cell orient for NC check PC
        #cut_sc = self.Cut_Supercell(Translate, lattice, repeat, crystal_sys)
        '''
        if crystal_sys == "hexagonal": ###
            hex_ref = self.Hex_ATS_Ref(cut_sc, lattice, repeat)
            cut_sc = self.Translate_Ref_to_000(cut_sc, hex_ref)
        ###
        for entry in cut_sc:
            point = entry[0]
            if -1 < np.around(point[0],3) <= 0:
                print(point)
        '''
        '''#print('Cut Cell',cut_sc)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for point in cut_sc:
            test = point[0]
            x = np.around(test[0],3)
            y = np.around(test[1],3)
            z = np.around(test[2],3)
            ax.scatter(x,y,z, c='k')
        plt.show()
        '''###
        #print(len(cut_sc))
        #TRef = self.Cut_SC_Trans_Ref(cut_sc, lattice, crystal_sys) #
        #Translate2 = self.Translate_Ref_to_000(cut_sc, TRef) #
        #print('T2',Translate2)
        '''###
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for point in Translate2:
            test = point[0]
            x = np.around(test[0],3)
            y = np.around(test[1],3)
            z = np.around(test[2],3)
            ax.scatter(x,y,z, c='k')
        plt.show()
        '''###
        #PRef = self.Select_Plane_Reference(cut_sc, lattice, miller, repeat)
        #print('pref',PRef)
        #print('translate2',Translate2)
        #POP = self.Get_All_Unique_Pts_on_Plane(
        #    cut_sc, PRef, sc_structure, miller) # MIM not most pref miler
        #MP = self.Translate_Ref_to_000(POP, PRef)
        '''###
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for point in MP:
            test = point[0]
            x = np.around(test[0],3)
            y = np.around(test[1],3)
            z = np.around(test[2],3)
            ax.scatter(x,y,z, c='k')
        plt.show()
        '''###
        #print('MP',MP)
        #Rep_Atoms = self.Pick_Rep_Atoms(MP, RAt, miller)
        #print('rep atoms',Rep_Atoms)
        #[RP, Rot_RA, RAZA, RAYA, RAXA] = self.Reorient_Plane(
        #    MP, Rep_Atoms, ELI, ELO)
        #fig = plt.figure()
        '''###
        #print('RP',RP)
        for test2 in RP:
            x2 = np.around(test2[0],3)
            y2 = np.around(test2[1],3)
            plt.scatter(x2,y2, c='k')
        plt.show()
        '''###
        #ZT = self.Get_Z_Trans_Length(lattice, miller, d_spacing, Rep_Atoms)
        #return RP, Rot_RA, RAZA, RAYA, RAXA, ZT
        return TR_NoE, RA_NoE, ZT

    def Reorient_Plane_W_Angles(self, points, RAZA, RAYA, RAXA, ELI, ELO):
        '''
        Rotates a set of points around an axis by a pre-calculated set of
        three angles for rotation around the three primary axes. This is done
        so that all the points have a z = 0 and the distances between the
        points are preserved.
        '''        
        SZ = 0
        RZ = False
        Align_to_X_Axis = self.Rotate(points, RAZA, SZ, RZ, ELI, ELO)
        
        SY = 1
        RY = False
        Rotate_Around_Y = self.Rotate(Align_to_X_Axis, RAYA,
                                      SY, RY, ELI, ELO)
            
        SX = 2
        RX = True
        Rotate_Around_X = self.Rotate(Rotate_Around_Y, RAXA,
                                      SX, RX, ELI, ELO)
        return Rotate_Around_X

    def GOMUC(self, structure, crystal_sys, miller, vecs, RE): #reorder GOMUC
        '''
        It takes the structure and match vectors and gives a rotated,
        3D crystal unit cell and rotated matching vectors. Also returns
        the d_spacing and reduced 2D unit cell vectors for x/y translations
        that occur later in the program.
        '''
        [RP, Rot_RA, ZT] = self.GOCP(structure, crystal_sys, miller,
                                     RE, ELI=True, ELO=False)
        TVs = [Rot_RA[0][0],Rot_RA[1][0]]

        max_miller = max(miller)
        repeat = (2*max_miller)+1 # make smart scale repeat
        lattice = structure.lattice
        d_spacing = self.Get_D_Spacing(miller, lattice, crystal_sys)

        slabgen = SlabGenerator(structure, miller, repeat, 3,
                                lll_reduce=True, primitive=False,
                                in_unit_planes=True)
        slabs = slabgen.get_slabs()
        Optimum_Slab_Face_Atoms_WE, OSP = self.Get_Slab_Face_with_RE(
            slabs, repeat, RE)
        Optimum_Slab = slabs[OSP]
        Optimum_Slab.make_supercell([repeat,repeat,1])
        OOS = reorient_z(Optimum_Slab)
        #print("OES slab matrix",OES._lattice.matrix)

        OS_sc_species = OOS.species # NEW
        OS_init_coords = OOS.cart_coords # NEW
        
        OS_coord_ele = self.Coords_Ele(OS_init_coords, OS_sc_species)
        
        MRef = self.Get_Middle_Of_Face_Atoms(Optimum_Slab_Face_Atoms_WE, RE)
        Rot_MSS = self.Translate_Ref_to_000(OS_coord_ele, MRef)
        
        miller_vec = np.array((lattice.matrix[0]*miller[0])+(
            lattice.matrix[1]*miller[1])+(
                lattice.matrix[2]*miller[2]))
        Rot_Vecs = [vecs[0],vecs[1]]
        
        #new_z = get_mi_vec(structure)
        #new_z = miller_vec/np.linalg.norm(miller_vec)
        #a, b, c = structure.lattice.matrix
        #new_x = a / np.linalg.norm(a)
        #new_y = np.cross(new_z, new_x)
        #x, y, z = np.eye(3)
        #rot_matrix = np.array([np.dot(*el) for el in itertools.product([x, y, z], [new_x, new_y, new_z])]).reshape(3, 3)
        #rot_matrix = np.transpose(rot_matrix)
        #RMV = np.dot(rot_matrix, miller_vec)
        RMV = [0,0,ZT]
        print("RMV",RMV)
        '''
        [RP, Rot_RA, RAZA, RAYA, RAXA, ZT] = self.GOCP(
            structure, crystal_sys, miller, ELI=True, ELO=False)
        TVs = [Rot_RA[0][0],Rot_RA[1][0]]
        #Red_UC = self.Get_Red_UC(RP, Rot_RA)
        
        max_miller = max(miller)
        repeat = (2*max_miller)+3 # make smart scale repeat
        lattice = structure.lattice
        d_spacing = self.Get_D_Spacing(miller, lattice, crystal_sys)
        
        MSS_Structure = structure * (repeat, repeat, repeat)
        MSS_Lattice = MSS_Structure.lattice
        MSS_Species = MSS_Structure.species
        MSS_IC = MSS_Structure.cart_coords

        MSS_CE = self.Coords_Ele(MSS_IC, MSS_Species)
        TCRef = self.Select_SC_Cut_Ref(MSS_CE, MSS_Lattice, crystal_sys)
        Translate = self.Translate_Ref_to_000(MSS_CE, TCRef)
        cut_sc = self.Cut_Supercell(Translate, lattice, repeat, crystal_sys)
        if crystal_sys == "hexagonal": ###
            hex_ref = self.Hex_ATS_Ref(cut_sc, lattice, repeat)
            cut_sc = self.Translate_Ref_to_000(cut_sc, hex_ref)
        PRef = self.Select_Plane_Reference(cut_sc, lattice, miller,
                                           repeat, TCRef)
        
        TCSS = self.Translate_Ref_to_000(cut_sc, PRef)
        ELI_MSS = True
        ELO_MSS = True
        Rot_MSS = self.Reorient_Plane_W_Angles(TCSS, RAZA, RAYA, RAXA,
                                               ELI_MSS, ELO_MSS)

        miller_vec = np.array((lattice.matrix[0]*miller[0])+(
            lattice.matrix[1]*miller[1])+(
                lattice.matrix[2]*miller[2]))
        Vecs = [vecs[0],vecs[1],miller_vec]
        ELI_vecs = False
        ELO_vecs = False
        Rot = self.Reorient_Plane_W_Angles(Vecs, RAZA, RAYA, RAXA,
                                                ELI_vecs, ELO_vecs)
        Rot_Vecs = [Rot[0],Rot[1]]
        RMV = Rot[2]
        #print('miller vec',miller_vec)
        #print('rot miller vec',RMV)
        '''
        return Rot_MSS, Rot_Vecs, TVs, d_spacing, ZT, RMV

class FilmAnalyzer:
    def __init__(self, film_max_miller=1,substrate_max_miller=1,
                 Convert_Trigonal_to_Hexagonal=False,
                 Only_Lowest_Strain_Per_Sub_Miller = False,
                 Use_Compliance_Tensor = True):
        self.Convert_Trigonal_to_Hexagonal = Convert_Trigonal_to_Hexagonal
        self.Only_Lowest_Strain_Per_Sub_Miller = Only_Lowest_Strain_Per_Sub_Miller
        self.film_max_miller = film_max_miller
        self.substrate_max_miller = substrate_max_miller
        self.Use_Compliance_Tensor = Use_Compliance_Tensor

    def Reverse_Reorient_Plane(self, points, RAZA, RAYA, RAXA):
        '''
        Using the angles determined in Reorient_Plane, takes a set of
        points and reveres the rotation done previously so that the points
        are in their original positions within the SC.
        '''
        gocp = Get_Oriented_Crystal_Plane()
        ELI = False
        ELO = False

        SX = 2
        RX = False
        Rotate_Around_X = gocp.Rotate(points, -1*RAXA, SX, RX, ELI, ELO)
        
        SY = 1
        RY = False
        Rotate_Around_Y = gocp.Rotate(Rotate_Around_X, -1*RAYA,
                                      SY, RY, ELI, ELO)     

        SZ = 0
        RZ = True
        Align_to_X_Axis = gocp.Rotate(Rotate_Around_Y, -1*RAZA,
                                      SZ, RZ, ELI, ELO)
        return Align_to_X_Axis

    def Generate_Surface_Vectors(self, millers, structure, crystal_sys, is_film):
        '''
        ##############
        '''
        OXS = structure.composition.oxi_state_guesses()
        ox_states = OXS[0]
        for species in structure.species:
            s_str = str(species)
            if is_film == True:
                max_miller = self.film_max_miller
                if ox_states[s_str] <= 0:
                    rep_element = s_str
                    break
            else:
                max_miller = self.substrate_max_miller
                if ox_states[s_str] >= 0:
                    rep_element = s_str
                    break
        
        #if is_film == True: OLD
        #    max_miller = self.film_max_miller OLD
        #else: OLD
        #    max_miller = self.substrate_max_miller OLD
        
        if millers is None:
            millers = sorted(
                get_symmetrically_distinct_miller_indices(
                    structure, max_miller))
        
        gocp = Get_Oriented_Crystal_Plane()
        vector_sets = []
        for m in millers:
            print('current miller',m)
            #[RP, Rot_RA, RAZA, RAYA, RAXA, ZT] = gocp.GOCP(
                #structure, crystal_sys, m, rep_element, ELI=True, ELO=False)
            [RP, Rot_RA, ZT] = gocp.GOCP(structure, crystal_sys, m,
                                 rep_element, ELI=True, ELO=False)
            Red_UC = gocp.Get_Red_UC(RP, Rot_RA)
            FFV = Red_UC
            #FFV = self.Reverse_Reorient_Plane(Red_UC, RAZA, RAYA, RAXA)
            print('FSV',FFV)
            vector_sets.append((FFV,m))
        #print("VS",vector_sets)
        return vector_sets
        
    def calculate(self, surface_vector_sets, lowest):
        '''
        Takes user given film and substrate structures, the structures'
        crystal systems, and user given millers and generates a match
        using the ZAM superlattice match principles. If the user has not
        given any specific millers of interest, the program takes the user
        given max miller indices and generates all unique miller planes
        up to the maximum miller given. The default max miller is one.
        '''
        self.zsl = ZSLGenerator(max_area_ratio_tol=0.15, max_area=600, ##
                                max_length_tol=0.08, max_angle_tol=0.01)
        #print('SEMI',get_symmetrically_equivalent_miller_indices(
            #self.substrate,substrate_millers[0]))
        
        for [film_vectors, substrate_vectors, film_miller,
             substrate_miller] in surface_vector_sets:
            for match in self.zsl(film_vectors, substrate_vectors, lowest):
                match['film_miller'] = film_miller
                match['sub_miller'] = substrate_miller
                yield match

    def Get_Length_and_FC(self, points, lattice, crystal_sys): ###
        '''
        Calculates the distance of the point from [0,0,0] when given
        fractional coordinates.
        '''
        CVWL = []
        for pt in points:
            distance = np.around(np.sqrt(pt[0]**2+pt[1]**2+pt[2]**2),3)
            pCvecs = lattice.get_fractional_coords(pt)
            Cvecs = [np.around(x,3) for x in pCvecs]
            cvwl = [Cvecs, distance]
            CVWL.append(cvwl)
        return CVWL

    def RemoveNegativeFromZero(self, vector):
        '''
        Takes a point and removes the negative from zero. Just for purely
        aesthetic reasons.
        '''
        for x in range(0, len(vector)):
            y = vector[x]
            if y == 0:
                y = 0.0
                vector[x] = y
            else:
                pass
        return vector

    def Compare_Values_in_Vec(self, vec):
        '''
        Compares the values within a vector. If all non-zero values are
        equal, the verdict remains true and takes the position of the first
        equal value. If they are not equal, returns a false verdict. Acts
        as a selector for the if/else statement in Get_Direction.
        '''
        verdict = True
        times = 0
        length = len(vec)
        absolute = [np.abs(v) for v in vec]
        zeros = tuple(absolute).count(0)
        if zeros == length-1:
            cheat = max(tuple(absolute))
            position = tuple(absolute).index(cheat)
        else:
            for place1 in range(0,length):
                c_start = place1+1
                num = vec[place1]
                if num == 0:
                    pass
                else:
                    for place2 in range(c_start,length):
                        num2 = vec[place2]
                        if num2 == 0:
                            pass
                        else:
                            if np.abs(num) == np.abs(num2):
                                if times == 0:
                                    position = place1
                                    times = times+1
                                else:
                                    pass
                            else:
                                verdict = False
                                position = 'NA'
                                return verdict, position
        return verdict, position
    
    def Mult_Int(self, vec): ## changed here
        '''
        Takes a vector and determines the minimum value necessary to make
        all values a whole number. If all components are whole, returns one.
        '''
        chop = [np.abs(vec[0]-int(vec[0])),np.abs(vec[1]-int(vec[1])),
                np.abs(vec[2]-int(vec[2]))]
        mults = []
        num = 1
        while num < 51:
            diffs = [np.around(num*x-np.around(num*x,0),2) for x in chop]
            if any(diffs) == False:
                return num
            else:
                num+=1

    def Get_Direction(self, vec, crystal_sys): # probably wrong
        '''
        Gives the most simplified, integer direction of an atom within
        the crystal structure. The direction reference is [0,0,0].
        '''
        if crystal_sys == "hexagonal": ###
            U = np.around((2*vec[1] - vec[0]),2)
            V = np.around((2*vec[0] - vec[1]),2)
            T = -1*(U + V)
            W = np.around(float(3*vec[2]),2)
            vec = [U,V,T,W]
            comp = [U-np.around(U,0),V-np.around(V,0),T-np.around(T,0),
                    W-np.around(W,0)]
        else:
            comp = [vec[0]-np.around(vec[0],0),vec[1]-np.around(vec[1],0),
                    vec[2]-np.around(vec[2],0)]
        if any(comp) == False:
            if crystal_sys == "hexagonal": ###
                GCD = gcd(int(np.around(vec[0],0)),int(np.around(vec[1],0)),
                          int(np.around(vec[2],0)),int(np.around(vec[3],0)))
            else:
                GCD = gcd(int(np.around(vec[0],0)),int(np.around(vec[1],0)),
                          int(np.around(vec[2],0)))
            direc = [int(np.around(x1/GCD,0)) for x1 in vec]
        else:
            [compare, pos] = self.Compare_Values_in_Vec(vec)
            if compare == True:
                direc = [int(np.around(x2/np.abs(vec[pos]),0)) for x2 in vec]
            else:
                mult = self.Mult_Int(vec)
                direc = [y3*mult for y3 in vec]
                if crystal_sys == "hexagonal": ###
                    GCD = gcd(int(np.around(direc[0],0)),
                              int(np.around(direc[1],0)),
                              int(np.around(direc[2],0)),
                              int(np.around(direc[3],0)))
                else:
                    GCD = gcd(int(np.around(direc[0],0)),
                              int(np.around(direc[1],0)),
                              int(np.around(direc[2],0)))
                direc = [int(np.around(x3/GCD,0)) for x3 in direc]
        return direc

    def Get_Match_Vectors_and_Directions(self, vecs, lattice, crystal_sys):
        '''
        Takes cartesian atomic positions and converts them to fractional
        coordinates, assigns them a distance from [0,0,0], removes zero
        from the point, and gets the direction of the atom within the
        crystal structure.
        '''
        CvecsWL = self.Get_Length_and_FC(
            vecs, lattice, crystal_sys) # coming out not as nice fractions
        for CvecWL in CvecsWL:
            self.RemoveNegativeFromZero(CvecWL[0])
        #sfa_direction = self.Get_Direction(CvecsWL[0][0], crystal_sys)
        sfa_direction = "TBA" # temp
        #sfb_direction = self.Get_Direction(CvecsWL[1][0], crystal_sys)
        sfb_direction = "TBA" # temp
        return CvecsWL, sfa_direction, sfb_direction

    def Direc_Cos(self, vec):
        '''
        Calculates the direction cosines for a given atomic position
        within a crystal structure.
        '''
        cosines = []
        for n in vec:
            value = n/(np.sqrt((vec[0]**2)+(vec[1]**2)+(vec[2]**2)))
            cosines.append(value)
        return cosines

    def Get_Strain_and_Strain_Energy(self, s, match_area, FWL, SWL):
        '''
        Assuming that the film is straining to the substrate, takes the
        two lattice match film vectors and calculates the film a to
        substrate a, and film b to substrate b strains. Then uses the
        direction cosines for the film a and film b to determine the direction
        dependent elastic complainces. Uses the strains and compliances to
        calculate the filma and film b stress which is then converted to a
        strain energy per unit volume. This volume is then converted to
        pascals and multiplied by the match area to get a thickness dependent
        strain energy (units = N) normalized to each individual match's match
        area.

        The process is based on a paper published by Jianyun Shen in 2002:
        INSERT REFERENCE HERE!
        '''
        Ls = self.Direc_Cos(FWL[0][0])
        Ms = self.Direc_Cos(FWL[1][0])
        
        Seg111 = ((Ls[0]**4)*s[0])+((Ls[1]**4)*s[6])+((Ls[2]**4)*s[11])
        Seg112 = ((Ls[0]**2)*(Ls[1]**2)*((2*s[1])+s[20]))+(
            (Ls[0]**2)*(Ls[2]**2)*((2*s[2])+s[18]))+(
                (Ls[1]**2)*(Ls[2]**2)*((2*s[7])+s[15]))
        Seg113 = (2*(Ls[0]**2)*Ls[1]*Ls[2]*(s[3]+s[19]))+(
            2*(Ls[1]**2)*Ls[0]*Ls[2]*(s[9]+s[17]))+(
                2*(Ls[2]**2)*Ls[1]*Ls[0]*(s[14]+s[16]))
        Seg114 = (2*(Ls[0]**3)*((Ls[2]*s[4])+(Ls[1]*s[5])))+(
            2*(Ls[1]**3)*((Ls[2]*s[8])+(Ls[0]*s[10])))+(
                2*(Ls[2]**3)*((Ls[1]*s[12])+(Ls[0]*s[13])))
        S11 = Seg111+Seg112+Seg113+Seg114
                
        Seg121 = (Ls[0]**2)*(((Ms[0]**2)*s[0])+((Ms[1]**2)*s[1])+(
            (Ms[2]**2)*s[2])+(Ms[1]*Ms[2]*s[3])+(Ms[0]*Ms[2]*s[4])+(
                Ms[0]*Ms[1]*s[5]))
        Seg122 = (Ls[1]**2)*(((Ms[0]**2)*s[1])+((Ms[1]**2)*s[6])+(
            (Ms[2]**2)*s[7])+(Ms[1]*Ms[2]*s[8])+(Ms[0]*Ms[2]*s[9])+(
                Ms[0]*Ms[1]*s[10]))
        Seg123 = (Ls[2]**2)*(((Ms[0]**2)*s[2])+((Ms[1]**2)*s[7])+(
            (Ms[2]**2)*s[11])+(Ms[1]*Ms[2]*s[12])+(Ms[0]*Ms[2]*s[13])+(
                Ms[0]*Ms[1]*s[14]))
        Seg124 = (Ls[1]*Ls[2])*(((Ms[0]**2)*s[3])+((Ms[1]**2)*s[8])+(
            (Ms[2]**2)*s[12])+(Ms[1]*Ms[2]*s[15])+(Ms[0]*Ms[2]*s[16])+(
                Ms[0]*Ms[1]*s[17]))
        Seg125 = (Ls[0]*Ls[2])*(((Ms[0]**2)*s[4])+((Ms[1]**2)*s[9])+(
            (Ms[2]**2)*s[13])+(Ms[1]*Ms[2]*s[16])+(Ms[0]*Ms[2]*s[18])+(
                Ms[0]*Ms[1]*s[19]))
        Seg126 = (Ls[0]*Ls[1])*(((Ms[0]**2)*s[5])+((Ms[1]**2)*s[10])+(
            (Ms[2]**2)*s[14])+(Ms[1]*Ms[2]*s[17])+(Ms[0]*Ms[2]*s[19])+(
                Ms[0]*Ms[1]*s[20]))
        S12 = Seg121+Seg122+Seg123+Seg124+Seg125+Seg126
                
        Seg221 = ((Ms[0]**4)*s[0])+((Ms[1]**4)*s[6])+((Ms[2]**4)*s[11])
        Seg222 = ((Ms[0]**2)*(Ms[1]**2)*((2*s[1])+s[20]))+(
            (Ms[0]**2)*(Ms[2]**2)*((2*s[2])+s[18]))+(
                (Ms[1]**2)*(Ms[2]**2)*((2*s[7])+s[15]))
        Seg223 = (2*(Ms[0]**2)*Ms[1]*Ms[2]*(s[3]+s[19]))+(
            2*(Ms[1]**2)*Ms[0]*Ms[2]*(s[9]+s[17]))+(
                2*(Ms[2]**2)*Ms[1]*Ms[0]*(s[14]+s[16]))
        Seg224 = (2*(Ms[0]**3)*((Ms[2]*s[4])+(Ms[1]*s[5])))+(
            2*(Ms[1]**3)*((Ms[2]*s[8])+(Ms[0]*s[10])))+(
                2*(Ms[2]**3)*((Ms[1]*s[12])+(Ms[0]*s[13])))
        S22 = Seg221+Seg222+Seg223+Seg224
                
        Strain_Fa = (SWL[0][1]-FWL[0][1])/(FWL[0][1])
        Strain_Fb = (SWL[1][1]-FWL[1][1])/(FWL[1][1])
        Stress_Fa = ((Strain_Fa*S22)-(Strain_Fb*S12))/((S11*S22)-(S12**2))
        Stress_Fb = ((Strain_Fb*S11)-(Strain_Fa*S12))/((S11*S22)-(S12**2))
        f_PE = 0.5*((Stress_Fa*Strain_Fa)+(Stress_Fb*Strain_Fb))
        SE_Pa = f_PE*(1*10**9)
        area_m = match_area*((1*10**(-10))**2)
        SE_N = SE_Pa*area_m
        return Strain_Fa, Strain_Fb, SE_N

    def ReduceElasTens(self, elas_tens):
        '''
        Takes a 6x6 elastic tensor matrix and removes the redundant values
        since the elastic tensor is symmetrc over the s11 to s66 diagonal.
        '''
        red_elas_tens = []
        take_cols = 0
        for row in range(0, 6):
            ETR = elas_tens[row]
            tens = ETR[take_cols:7]
            red_elas_tens.extend(tens)
            take_cols = take_cols+1
        return red_elas_tens

    def RemoveRedundantVectors(self, df, sn, fn):
        '''
        Takes the full matching vectors Pandas dataframe and simplifies
        the entries to one representative per given area. Selects the entry
        with the lowest strain, and shortest substrate a vector.
        '''
        Count = 0
        count = 0
        MC = 50
        mc = 14 ### change # of accpeted entries per substrate/film pair regardless of ori
        all_matches_revised = []
        a1 = df.iloc[0]
        mps1 = [a1['film id'], a1['sub id']]
        area1 = a1['area']
        sa1 = a1['sa']
        fa1 = a1['fa']
        for num in range(0,len(df)):
            a2 = df.iloc[num]
            mps2 = [a2['film id'], a2['sub id']]
            area2 = a2['area']
            sa2 = a2['sa']
            fa2 = a2['fa']
            if sn + fn == 2:
                if Count <= MC:
                    if num == 0:
                        all_matches_revised.append(a1)
                        Count = Count + 1
                    else:
                        if np.around(area1,2) != np.around(area2,2): ### rounding
                            all_matches_revised.append(a2)
                            area1 = area2
                            sa1 = sa2
                            fa1 = fa2
                            Count = Count + 1
                        else:
                            if np.around(sa1,3) != np.around(sa2,3): ### rounding
                                all_matches_revised.append(a2)
                                area1 = area2
                                sa1 = sa2
                                fa1 = fa2
                                Count = Count + 1
                            else:
                                if np.around(fa1,3) != np.around(fa2,3):
                                    all_matches_revised.append(a2)
                                    area1 = area2
                                    sa1 = sa2
                                    fa1 = fa2
                                    Count = Count + 1
            else:
                if count <= mc:
                    if num == 0:
                        all_matches_revised.append(a1)
                        count = count + 1
                    elif mps1[0] == mps2[0] and mps1[1] == mps2[1]:
                        if np.around(area1,2) != np.around(area2,2): ### rounding
                            all_matches_revised.append(a2)
                            area1 = area2
                            sa1 = sa2
                            fa1 = fa2
                            mps1 = mps2
                            count = count + 1
                        else:
                            if np.around(sa1,3) != np.around(sa2,3): ### rounding
                                all_matches_revised.append(a2)
                                area1 = area2
                                sa1 = sa2
                                fa1 = fa2
                                mps1 = mps2
                                count = count + 1
                            else:
                                if np.around(fa1,3) != np.around(fa2,3):
                                    all_matches_revised.append(a2)
                                    area1 = area2
                                    sa1 = sa2
                                    fa1 = fa2
                                    mps1 = mps2
                                    count = count + 1
                    else:
                        all_matches_revised.append(a2)
                        #print('reset lt mc', mps1[0], mps1[1], mps2[0], mps2[1])
                        area1 = area2
                        sa1 = sa2
                        fa1 = fa2
                        mps1 = mps2
                        count = 1
                elif count > mc and (mps1[0] != mps2[0] or mps1[1] != mps2[1]):
                    all_matches_revised.append(a2)
                    #print('reset gt mc', mps1[0], mps1[1], mps2[0], mps2[1])
                    area1 = area2
                    sa1 = sa2
                    fa1 = fa2
                    mps1 = mps2
                    count = 1
        #print('am len', len(all_matches_revised))
        return all_matches_revised

    def Simp_Matches(self, df):
        One_FM_List = []
        Face_Matches = []
        a1 = df.iloc[0]
        fm1 = [a1['film id'], a1['sub id'], a1['f orient'], a1['s orient']]
        for num in range(0, len(df)):
            a2 = df.iloc[num]
            fm2 = [a2['film id'], a2['sub id'], a2['f orient'], a2['s orient']]
            fm2e = [a2['sub id'], a2['film id'], a2['s orient'], a2['f orient']]
            if num == 0:
                Face_Matches.append(fm1)
                One_FM_List.append(a1)
                #print('mps initiate')
            else:
                if (fm2 not in Face_Matches) and (fm2e not in Face_Matches):
                    One_FM_List.append(a2)
                    Face_Matches.append(fm2)
                    #print('mps changed')
        return One_FM_List

    def Make_Comp_DF(self, DF):
        '''
        Takes the full matching dataframe and only includes the parameters
        needed to directly compare the epitaxial matches.
        '''
        NDF = []
        for num in range(0,len(DF)):
            item = DF.iloc[num]
            db_entry = {
                "ff": item["ff"],
                "f sg": item["f sg"],
                "f orient": item["f orient"],
                "sf": item["sf"],
                "s sg": item["s sg"],
                "s orient": item["s orient"],
                "fa direction": item["fa direction"],
                "fb direction": item["fb direction"],
                "sa direction": item["sa direction"],
                "sb direction": item["sb direction"],
                "area": item["area"],
                "fa": item["fa"],
                "fb": item["fb"],
                "f alpha": item["f alpha"],
                "sa": item["sa"],
                "sb": item["sb"],
                "s alpha": item["s alpha"],
                "fa strain": item["fa strain"],
                "fb strain": item["fb strain"],
                "Strain Energy (N)": item["Strain Energy (N)"],
                "FOM": item["FOM"]
                }
            NDF.append(db_entry)
        return NDF

    def Make_Imaging_DF(self, DF):
        '''
        Takes the full matching dataframe and only includes the parameters
        needed to image the matches. Not intended for viewing by the user.
        '''
        NIDF = []
        for num in range(0,len(DF)):
            item = DF.iloc[num]
            db_entry = {
                "film id": item["film id"],
                "ff": item["ff"],
                "f orient": item["f orient"],
                "sub id": item["sub id"],
                "sf": item["sf"],
                "s orient": item["s orient"],
                "fa vector": item["fa vector"],
                "fb vector": item["fb vector"],
                "sa vector": item["sa vector"],
                "sb vector": item["sb vector"]
                }
            NIDF.append(db_entry)
        return NIDF

    def Simplify_Miller_List(self, millers, struct):### check here
        simplified = []
        reject = []
        if len(millers) > 1:
            for miller in millers:
                if miller not in simplified and miller not in reject:
                    SEMI = get_symmetrically_equivalent_miller_indices(
                        struct, miller, return_hkil=False)
                    #print('SEMI',SEMI)
                    equiv = []
                    equiv.append(miller)
                    for other in millers:
                        if other in SEMI and other not in equiv:
                            equiv.append(other)
                    if len(equiv) == 1:
                        simplified.append(equiv[0])
                    else:
                        ref1 = equiv[0]
                        for num in range(1,len(equiv)):
                            ref2 = equiv[num]
                            if ref1.count(0) < ref2.count(0): # changed > to < cuz we want more 0's
                                reject.append(ref1)
                                ref2 = ref1
                            elif ref1.count(0) == ref2.count(0):
                                if (sum(map(lambda x : x < 0, ref1)) >
                                    sum(map(lambda x : x < 0, ref2))):
                                    reject.append(ref1)
                                    ref1 = ref2
                                elif (sum(map(lambda x : x < 0, ref1)) ==
                                      sum(map(lambda x : x < 0, ref2))):
                                    if ref1.count(0) == 2:
                                        if (tuple(np.abs(ref1)).index(
                                            max(np.abs(ref1))) >
                                            tuple(np.abs(ref2)).index(
                                                max(np.abs(ref1)))):
                                            reject.append(ref1)
                                            ref1 = ref2
                                        else:
                                            reject.append(ref2)
                                    elif ref1.count(0) == 1:
                                        if sum(map(lambda x : x < 0, ref1)) == 1:
                                            if ref1.index(0) > ref2.index(0):
                                                reject.append(ref1)
                                                ref1 = ref2
                                            elif ref1.index(0) == ref2.index(0):
                                                if (ref1.index(max(ref1)) >
                                                    ref2.index(max(ref2))):
                                                    reject.append(ref1)
                                                    ref1 = ref2
                                                else:
                                                    reject.append(ref2)
                                            else:
                                                reject.append(ref2)
                                        else:
                                            if ref1.index(0) < ref2.index(0):
                                                reject.append(ref1)
                                                ref1 = ref2
                                            else:
                                                reject.append(ref2)
                                    else:
                                        if sum(map(lambda x : x < 0, ref1)) == 1:
                                            if (ref1.index(min(ref1)) <
                                                ref2.index(min(ref2))):
                                                reject.append(ref1)
                                                ref1 = ref2
                                            else:
                                                reject.append(ref2)
                                        else:
                                            if (ref1.index(max(ref1)) >
                                                ref2.index(max(ref2))):
                                                reject.append(ref1)
                                                ref1 = ref2
                                            else:
                                                reject.append(ref2)
                                else:
                                    reject.append(ref2)
                        simplified.append(ref1)
        else:
            simplified = millers
        print('simplified millers',simplified)
        #print('reject',reject)
        return simplified

    def Execute_Film_Matcher(self, films, substrates,
                             film_millers, substrate_millers):
        '''
        Takes the user given inputs and executes the film matcher program.
        It gives three dataframes as excel files. The first includes ALL
        match vectors and directions, the second contains a list of simplified
        matches where only one match with the lowest strain energy per area
        is included, and the third contains all data needed to image the
        matches. The imaging excel file is not intended to be viewed.
        '''
        all_matches = []

        for s in substrates:
            sub_num = len(substrates)
            sub_id = s['material_id']
            print('sub id',sub_id)
            init_substrate = s["structure"]
            sga1 = SpacegroupAnalyzer(init_substrate)
            s_SpGr = sga1.get_space_group_symbol()
            iscrystal_sys = sga1.get_crystal_system()
            if iscrystal_sys == "trigonal":
                if self.Convert_Trigonal_to_Hexagonal == True:
                    substrate = sga1.get_conventional_standard_structure()
                    scrystal_sys = "hexagonal"
                else:
                    substrate = init_substrate
                    scrystal_sys = iscrystal_sys
            else:
                substrate = sga1.get_conventional_standard_structure()
                scrystal_sys = iscrystal_sys

            # def here to simplify substrate miller list
            Substrate_millers = self.Simplify_Miller_List(
                    substrate_millers, substrate)
            print('sub millers',Substrate_millers)

            IF1 = False
            SFVs = self.Generate_Surface_Vectors(
                Substrate_millers, substrate, scrystal_sys, IF1)
                
            for f in tqdm(films):
                film_num = len(films)
                film_id = f['material_id']
                print("FID",film_id)
                init_film = f["structure"]
                print('film',init_film)
                sga2 = SpacegroupAnalyzer(init_film)
                f_SpGr = sga2.get_space_group_symbol()
                ifcrystal_sys = sga2.get_crystal_system()
                if ifcrystal_sys == "trigonal":
                    if self.Convert_Trigonal_to_Hexagonal == True:
                        film = sga2.get_conventional_standard_structure()
                        fcrystal_sys = "hexagonal"
                    else:
                        film = init_film
                        fcrystal_sys = ifcrystal_sys
                else:
                    film = sga2.get_conventional_standard_structure()
                    fcrystal_sys = ifcrystal_sys

                print('film',film)
                    
                elas_tens = f["elasticity.compliance_tensor"]
                try:
                    RET = self.ReduceElasTens(elas_tens)
                    s = [x/1000 for x in RET]
                except:
                    f_strain = "NA"
                else:
                    f_strain = "Ok"

                # Def here to simplify film miller list
                Film_millers = self.Simplify_Miller_List(
                    film_millers, film)
                print('film millers',Film_millers)
                
                IF2 = True
                FFVs = self.Generate_Surface_Vectors(
                    Film_millers, film, fcrystal_sys, IF2)

                surface_vector_sets = []
                for SVs in SFVs:
                    for FVs in FFVs:
                        SVS_Entry = [FVs[0], SVs[0], FVs[1], SVs[1]]
                        surface_vector_sets.append(SVS_Entry)
                print("SVS",surface_vector_sets)
                
                matches_by_orient = self.calculate(surface_vector_sets,
                                                   lowest=False)
                print("out of ZAM")
                # CALCULATE IS NOW JUST GETTING ZSL MATCH LOOP
                
                for match in matches_by_orient:
                    slattice = substrate.lattice
                    svecs = match['sub_sl_vecs']
                    s_alpha = np.around(Vec_Angle_Rad(
                        svecs[0], svecs[1])*(180/np.pi),2)
                    new_match_area = np.around(vec_area(*svecs),2) ###

                    [CsvecsWL, sa_direction, sb_direction] = self.Get_Match_Vectors_and_Directions(
                        svecs, slattice, scrystal_sys)

                    flattice = film.lattice
                    fvecs = match['film_sl_vecs']
                    f_alpha = np.around(Vec_Angle_Rad(
                        fvecs[0], fvecs[1])*(180/np.pi),2)

                    [CfvecsWL, fa_direction, fb_direction] = self.Get_Match_Vectors_and_Directions(
                        fvecs, flattice, fcrystal_sys)

                    if f_strain == "NA":
                        Strain_Fa = (CsvecsWL[0][1]-CfvecsWL[0][1])/(CfvecsWL[0][1])
                        Strain_Fb = (CsvecsWL[1][1]-CfvecsWL[1][1])/(CfvecsWL[1][1])
                        SE_N = "No Compliance Tensor in MPD"
                        FOM = np.abs((new_match_area**2)*Strain_Fa*Strain_Fb)
                    else:
                        [Strain_Fa, Strain_Fb, SE_N] = self.Get_Strain_and_Strain_Energy(
                            s, new_match_area, CfvecsWL, CsvecsWL)
                        if self.Use_Compliance_Tensor == True:
                            FOM = np.abs((new_match_area**2)*SE_N)
                        else:
                            FOM = np.abs((new_match_area**2)*Strain_Fa*Strain_Fb)
                
                    db_entry = {
                        "film id": film_id,
                        "ff": film.composition.reduced_formula,
                        "f sg": f_SpGr,
                        "f orient": match["film_miller"],
                        "sub id": sub_id,
                        "sf": substrate.composition.reduced_formula,
                        "s sg": s_SpGr,
                        "s orient": match["sub_miller"],
                        "area": new_match_area,
                        "fa vector": CfvecsWL[0][0],
                        "fa direction": fa_direction,
                        "fb vector": CfvecsWL[1][0],
                        "fb direction": fb_direction,
                        "fa": np.around(CfvecsWL[0][1],3),
                        "fb": np.around(CfvecsWL[1][1],3),
                        "f alpha": f_alpha,
                        "sa vector": CsvecsWL[0][0],
                        "sa direction": sa_direction,
                        "sb vector": CsvecsWL[1][0],
                        "sb direction": sb_direction,
                        "sa": np.around(CsvecsWL[0][1],3),
                        "sb": np.around(CsvecsWL[1][1],3),
                        "s alpha": s_alpha,
                        "fa strain": np.around(Strain_Fa,4),
                        "fb strain": np.around(Strain_Fb,4),
                        "Strain Energy (N)": SE_N,
                        "FOM": FOM
                        }
                    if db_entry == []:
                        pass
                    else:
                        all_matches.append(db_entry)

        if all_matches == []: # error calling pretty form
            #no_match_string = "No matches for {} on {} have been found for the given miller indices.".format(
            #    s['pretty_formula'], f['pretty_formula'])
            no_match_string = "These faces are incompatible."
            print(no_match_string)
        else:
            print("in DF Sorting")
            sorted_all_matches = sorted(all_matches,key=itemgetter(
                'sub id','film id','area','sa','fa','FOM'))
            df = pd.DataFrame(sorted_all_matches)
            NCDF = self.Make_Comp_DF(df)
            ndf = pd.DataFrame(NCDF)
            AMR = self.RemoveRedundantVectors(df, sub_num, film_num) # maybe remove extra sort from
            # remove redundant vectors! Not sorted by FOM first so takes lowest area lowest FOM
            sorted_AMR = sorted(AMR,key=itemgetter('FOM','area','sa','fa'))
            if self.Only_Lowest_Strain_Per_Sub_Miller == True:
                SAMR_DF = pd.DataFrame(sorted_AMR)
                df2 = pd.DataFrame(self.Simp_Matches(SAMR_DF))
            else:
                df2 = pd.DataFrame(sorted_AMR)
            df2.reset_index(drop=True, inplace=True)
            NCDF2 = self.Make_Comp_DF(df2)
            ndf2 = pd.DataFrame(NCDF2)
            MDF = self.Make_Imaging_DF(df2)
            mdf = pd.DataFrame(MDF)

            ndf.to_csv('all_direc.csv')
            ndf2.to_csv('simp_direc.csv')
            mdf.to_csv('matching_df.csv')

class Image_Matches:
    '''
    Takes user inputs and the matching_df generated by the FilmAnalyzer
    program and generates XYZ files of the 3D match. The user can determine
    the matching lattice's length, width, and depth for both the film and
    the substrate, although the only independently selected variable is for
    the film and substrate thicknesses.
    '''
    def __init__(self, DF, matches=[0], py_image=False, slab_half_length=10,
                 slab_half_width=10, sub_slab_depth=0, film_slab_depth=0,
                 angstroms=False, strain_film=True,
                 Trigonal_Film_is_Hexagonal=False,
                 Symmetric_20nm_Chunk = False):
        self.DF = DF
        self.ms = matches
        self.PI = py_image
        self.shl = slab_half_length
        self.shw = slab_half_width
        self.ssd = sub_slab_depth
        self.fsd = film_slab_depth
        self.ang = angstroms
        self.sf = strain_film
        self.TFH = Trigonal_Film_is_Hexagonal
        self.S20C = Symmetric_20nm_Chunk

    def Get_Match_Info(self, match, D, FoS):
        '''
        Takes the data from the excel file generated by the FilmMatcher
        program, converts it into a usable format if needed, and assigns
        it a variable so that they can be called later.
        '''
        info = D[0]
        struct = info["structure"]
        SGA = SpacegroupAnalyzer(struct)
        crystal_sys = SGA.get_crystal_system()
        if crystal_sys in ["monoclinic", "trigonal"]: ###
            if self.TFH == True and crystal_sys == "trigonal":
                crystal_sys = "hexagonal"
                struct = SGA.get_conventional_standard_structure()
        else:
            struct = SGA.get_conventional_standard_structure()
        #print('FM CS',crystal_sys)
        if FoS == "F":
            form = match["ff"]
            split_miller = match['f orient'].split(' ')
            miller = (int(split_miller[0][1:-1]),
                      int(split_miller[1][0:-1]),
                      int(split_miller[2][0:-1]))
            vecs = [np.array(match['fa vector']),
                    np.array(match['fb vector'])]
            split_a_vec = match['fa vector'].split(' ')
            a_array = np.array([float(split_a_vec[0][1:-1]),
                                float(split_a_vec[1][0:-1]),
                                float(split_a_vec[2][0:-1])])
            split_b_vec = match['fb vector'].split(' ')
            b_array = np.array([float(split_b_vec[0][1:-1]),
                                float(split_b_vec[1][0:-1]),
                                float(split_b_vec[2][0:-1])])
            vecs = [a_array, b_array]
        else:
            form = match["sf"]
            split_miller = match['s orient'].split(' ')
            miller = (int(split_miller[0][1:-1]),
                      int(split_miller[1][0:-1]),
                      int(split_miller[2][0:-1]))
            vecs = [np.array(match['sa vector']),
                    np.array(match['sb vector'])]
            split_a_vec = match['sa vector'].split(' ')
            a_array = np.array([float(split_a_vec[0][1:-1]),
                                float(split_a_vec[1][0:-1]),
                                float(split_a_vec[2][0:-1])])
            split_b_vec = match['sb vector'].split(' ')
            b_array = np.array([float(split_b_vec[0][1:-1]),
                                float(split_b_vec[1][0:-1]),
                                float(split_b_vec[2][0:-1])])
            vecs = [a_array, b_array]
        return miller, struct, crystal_sys, vecs, form
  
    def Find_000_in_Match_Lattice(self, pointsWE):
        '''
        Takes a list of [coordinates and elements] and finds the entry (atom)
        that occupies the [0, 0, 0] position in the lattice.
        '''
        zero = tuple([0,0,0])
        for val in range(0, len(pointsWE)):
            PWE = pointsWE[val]
            P = tuple([np.around(np.abs(num),2) for num in PWE[0]])
            if zero == P:
                position = val
                return position

    def Reorder_2DUC_Vecs(self, UCV):
        '''
        Takes the two 2D unit cell vectors and reorders them so that the
        entry with the largest x-value is in the first position.
        '''
        if np.abs(UCV[0][0]) >= np.abs(UCV[1][0]):
            new_UCV = UCV
        else:
            new_UCV = [UCV[1], UCV[0]]
        return new_UCV

    def Get_Disp_Angle(self, vec1, vec2):
        '''
        Uses a set of selectors to determine which rotation angle should be
        used to align the lattices such that their match vectors are
        coincident and one match vector is on the x-axis. Provides a standard
        position that can be used to easily strain the lattice if desired.
        '''
        unit_v = [p / fast_norm(vec1) for p in vec1]
        DA1_rad = vec_angle(np.array([1,0,0]), unit_v)
        DA1 = np.around(DA1_rad*(180/np.pi),3)
        unit_v2 = [p / fast_norm(vec2) for p in vec2]
        DA2_rad = vec_angle(np.array([1,0,0]), unit_v2)
        DA2 = np.around(DA2_rad*(180/np.pi),3)
        if (DA1 <= 90 and DA2 <= 90 and vec1[1] >= 0 and vec2[1] >= 0) or (
            DA1 >= 90 and DA2 >= 90 and vec1[1] >= 0 and vec2[1] >= 0) or (
                (DA1 <= 90 or DA2 <= 90) and (DA1 >= 90 or DA2 >= 90) and vec1[
                    1] >= 0 and vec2[1] >= 0):
            if DA1 <= DA2:
                angle = -1*DA1_rad
            else:
                angle = -1*DA2_rad
        elif (DA1 >= 90 and DA2 >= 90 and (vec1[1] >= 0 or vec2[1] >= 0) and (
            vec1[1] <= 0 or vec2[1] <= 0)):
            if vec1[1] >= 0:
                angle = -1*DA1_rad
            else:
                angle = -1*DA2_rad
        elif (DA1 <= 90 and DA2 <= 90 and (vec1[1] >= 0 or vec2[1] >= 0) and (
            vec1[1] <= 0 or vec2[1] <= 0)):
            if vec1[1] >= 0:
                angle = DA2_rad
            else:
                angle = DA1_rad
        else:
            if DA1 >= DA2:
                angle = DA1_rad
            else:
                angle = DA2_rad
        return angle

    def Get_Partner_Rot_Angle(self, vec):
        '''
        Takes a vector and gets its rotation angle with respect to the [1,0,0]
        direction. Negative denotes counter clockwise and positive denotes
        clockwise rotation.
        '''
        unit_v = [p / fast_norm(vec) for p in vec]
        angle = np.around(vec_angle(np.array(
            [1,0,0]), unit_v)*(180/np.pi),3)
        if vec[1] <= 0:
            final_angle = angle
        else:
            final_angle = -1*angle
        return final_angle

    def Reorder_Vectors(self, vector_set): # changed
        '''
        Reorders the vectors post Disp_Angle selection so that the [x, 0, 0]
        vector is in the first position. For standardization purposes.
        '''
        vec1 = [np.around(v1,8) for v1 in vector_set[0]]
        vec2 = [np.around(v2,8) for v2 in vector_set[1]]
        check = [np.around(v3,3) for v3 in vec1]
        cheat = max(tuple(np.abs(check)))
        if tuple(check).count(0) == 2 and tuple(np.abs(
            check)).index(cheat) == 0:
            reordered_vec = [np.array(vec1), np.array(vec2)]
        else:
            reordered_vec = [np.array(vec2), np.array(vec1)]
        return reordered_vec

    def Get_Partner_Angle(self, SRVs, FMVs): ####
        #max_length_tol=0.14, max_angle_tol=0.01
        comp_svecs = []
        for svec in SRVs:
            distance = np.around(np.sqrt(svec[0]**2+svec[1]**2+svec[2]**2),3)
            angle = -1*self.Get_Partner_Rot_Angle(svec)
            comp = [svec, distance, angle]
            comp_svecs.append(comp)
        #print('comp_svecs',comp_svecs)

        test_angs = [self.Get_Partner_Rot_Angle(FMVs[0])*(
            np.pi/180), self.Get_Partner_Rot_Angle(FMVs[1])*(np.pi/180)]
        #print('test angles', test_angs)
        gocp = Get_Oriented_Crystal_Plane()
        for tangle in test_angs:
            test_vecs = []
            for trfv in FMVs:
                test_rotate = gocp.Rotate_Around_Z(trfv, tangle)
                test_vecs.append(test_rotate)
            final_test_vecs = self.Reorder_Vectors(test_vecs) # not reordering
            #print('FTV',final_test_vecs)
            comp_ftv = []
            for ftv in final_test_vecs:
                distance2 = np.around(np.sqrt(ftv[0]**2+ftv[1]**2+ftv[2]**2),3)
                angle2 = -1*self.Get_Partner_Rot_Angle(ftv)
                comp2 = [ftv, distance2, angle2]
                comp_ftv.append(comp2)
            #print('comp ftv',comp_ftv)
            if np.around(comp_ftv[0][1],3) == np.around(comp_ftv[1][1],3):
                #print('comp2',np.abs(comp_svecs[1][2]/comp_ftv[1][2] - 1))
                if (np.abs(np.abs(
                    comp_svecs[1][1]/comp_ftv[1][1]) - 1) <= 0.14) and (
                        np.abs(#np.abs(
                            comp_svecs[1][2]/comp_ftv[1][2] - 1) <= 0.01):
                    return tangle
            else:
                if (np.abs(np.abs(
                    comp_svecs[1][1]/comp_ftv[1][1]) - 1) <= 0.14) and (
                        np.abs(np.abs(
                            comp_svecs[1][2]/comp_ftv[1][2]) - 1) <= 0.01):
                    return tangle

    def Get_Rotated_Match_Vectors(self, SMVs, FMVs):
        #print('SMV FMV',SMVs,FMVs)
        '''
        Rotates the match vectors by a determined displacement angle and
        also returns the displacement angle.
        '''
        s_disp_angle = self.Get_Disp_Angle(SMVs[0], SMVs[1])
        gocp = Get_Oriented_Crystal_Plane()
        srot_vecs_cc = []
        for svec in SMVs:
            srot_cc = gocp.Rotate_Around_Z(svec, s_disp_angle)
            srot_vecs_cc.append(srot_cc)
        final_svecs_cc = self.Reorder_Vectors(srot_vecs_cc)
        f_disp_angle = self.Get_Partner_Angle(final_svecs_cc, FMVs)
        frot_vecs_cc = []
        for fvec in FMVs:
            frot_cc = gocp.Rotate_Around_Z(fvec, f_disp_angle)
            frot_vecs_cc.append(frot_cc)
        final_fvecs_cc = self.Reorder_Vectors(frot_vecs_cc)
        return final_svecs_cc, s_disp_angle, final_fvecs_cc, f_disp_angle

    def Make_Oriented_Matching_Lattice(self, points, vecs_cc, RRA,
                                       SMX, SMY, DA, Z_trans, RMV, is_film):
        print("VCC",vecs_cc, "RRA",RRA, "SMX",SMX, "SMY",SMY,
              "DA",DA, "ZT",Z_trans, "RMV",RMV)
        '''
        Makes an extended lattice in the x and y dimensions. The periodicity
        of the crystal face is preserved by using the rotated characteristic
        reference atoms to translate the atoms on the face within a given area.
        '''
        RUCV = self.Reorder_2DUC_Vecs(RRA)
        max_length = (SMX+SMY)/np.sqrt(2)
        x_range = int(np.ceil(max_length/np.abs(RUCV[0][0])))
        y_range = int(np.ceil(max_length/np.abs(RUCV[1][1])))

        EL = []
        strings = []
        Copy = copy.deepcopy(points)
        for n in range((-1*y_range),y_range+1):
            y_trans = n*RUCV[1] 
            for m in range((-1*x_range),x_range+1):
                x_trans = m*RUCV[0]
                tcopy = []
                for p in Copy:
                    P = p[0]
                    NP = [P+y_trans+x_trans, p[1]]
                    tcopy.append(NP)
                for point in tcopy:
                    rpoint = [np.around(x,8) for x in point[0]]
                    rtup = [np.around(x,3) for x in point[0]]
                    if str(rtup) not in strings:
                        strings.append(str(rtup))
                        rpointWE = [rpoint, point[1]]
                        EL.append(rpointWE)
    
        if self.ang == False:
            if is_film == False:
                MZ = Z_trans*np.abs(self.ssd)
                if self.S20C == True:
                    n = 2
                    while True:
                        if MZ < 10:
                            MZ = n*Z_trans*np.abs(self.ssd)
                            n += 1
                        else:
                            break
            else:
                MZ = Z_trans*np.abs(self.fsd)
                if self.S20C == True:
                    n = 2
                    while True:
                        if MZ < 10:
                            MZ = n*Z_trans*np.abs(self.fsd)
                            n += 1
                        else:
                            break
        else:
            if is_film == False:
                MZ = np.abs(self.ssd)
            else:
                MZ = np.abs(self.fsd)
        z_range = int(np.ceil(MZ/Z_trans))
        if is_film == True:
            #DA = DA+(60*(np.pi/180)) ### here for rot
            range_min = -1
            range_max = z_range + 1
            chunk_min = -1*0.01
            chunk_max = MZ + 0.01
        else:
            range_min = -1*(z_range+1)
            range_max = 1
            chunk_min = (-1*MZ) - 0.01
            chunk_max = 0.01
        TEL = []
        z_strings = []
        Copy2 = copy.deepcopy(EL)
        for n in range(range_min,range_max):
            z_trans = np.array([n*z for z in RMV]) ###
            #print('zt',z_trans)
            tcopy2 = []
            for p2 in Copy2:
                P2 = p2[0]
                NP2 = [P2+z_trans, p2[1]] ###
                tcopy2.append(NP2)
            for point2 in tcopy2:
                rpoint2 = [np.around(x2,7) for x2 in point2[0]]
                rtup2 = [np.around(x2,3) for x2 in point2[0]]
                if chunk_min <= rtup2[2] <= chunk_max:
                    if str(rtup2) in z_strings:
                        pass
                    else:
                        z_strings.append(str(rtup2))
                        rpointWE2 = [np.array(rpoint2), point2[1]]
                        TEL.append(rpointWE2)
                else:
                    pass
        gocp = Get_Oriented_Crystal_Plane()
        rot_pts = []
        for pointWE in TEL:
            point = pointWE[0]
            rot_point = gocp.Rotate_Around_Z(point, DA) ###
            rot_pointWE = [rot_point, pointWE[1]]
            rot_pts.append(rot_pointWE)
        return rot_pts

    def Cut_and_Strain_Lattice(self, pts, z_disp, x_strain, y_strain,
                               SMX, SMY, is_film):
        '''
        Uses a "displacement vector" to strain and raise the film lattice
        by an arbitrary distance calculated by adding the zero atom's radius
        for both the film and substrate and an extra constant.
        '''
        if is_film == True:
            if self.sf == True:
                disp_vec = [x_strain, y_strain, z_disp]
            else:
                disp_vec = [0, 0, z_disp]
        else:
            disp_vec = [0, 0, 0]
        final_lattice = []
        for ptWE in pts:
            pt = ptWE[0]
            TP = [pt[0]*disp_vec[0]+pt[0], pt[1]*disp_vec[1]+pt[1],
                  pt[2]+disp_vec[2]]
            RTP = [np.around(x,3) for x in TP]
            if (((-1*SMX)-.01) <= RTP[0] <= (SMX+.01) and (
                (-1*SMY)-.01) <= RTP[1] <= (SMY+.01)):
                PWE = [ptWE[1], RTP]
                final_lattice.append(PWE)
        return final_lattice

    def Make_Python_Image_Sub(self, pts):
        '''
        Takes the final substrate match atoms and takes only the surface
        layer for 2D match imaging.
        '''
        x = []
        y = []
        for v in pts:
            q = v[1]
            if q[2] == 0:
                x.append(q[0])
                y.append(q[1])
        return x, y

    def Make_Python_Image_Film(self, pts, z_disp):
        '''
        Takes the final film match atoms and takes only the surface
        layer for 2D match imaging.
        '''
        x = []
        y = []
        for v in pts:
            q = v[1]
            if z_disp-0.01 <= q[2] <= z_disp+0.01:
                x.append(q[0])
                y.append(q[1])
        return x, y
        
    def Get_Match_xyz_Files(self):
        '''
        Takes the data generated by the FilmMatcher program and makes a
        match XYZ file based on user given parameters such as: the matches
        to image; 3D crystal area given by a half width, half length, and
        a crystal depth which can be given as scalar multiples of the match
        length or in angstroms; and whether to strain the film to the
        substrate or save a python generated 2D match image.
        '''
        mpr = MPRester()
        for num in self.ms:
            match = self.DF.iloc[num]
            film_id = match["film id"]
            sub_id = match["sub id"]

            FoS_F = "F"
            F = mpr.query({"material_id": film_id}, ['structure'])
            [f_miller, Film, f_crystal_sys,
             f_vecs, f_form] = self.Get_Match_Info(match, F, FoS_F)
            f_lattice = Film.lattice
            S = mpr.query({"material_id": sub_id}, ['structure'])
            FoS_S = "S"
            [s_miller, Substrate, s_crystal_sys,
             s_vecs, s_form] = self.Get_Match_Info(match, S, FoS_S)
            s_lattice = Substrate.lattice
            F_vecs_cc = f_lattice.get_cartesian_coords(f_vecs)
            S_vecs_cc = s_lattice.get_cartesian_coords(s_vecs)
            #print('FVCC SVCC', F_vecs_cc,S_vecs_cc)
            #print('FCS SCS',f_crystal_sys,s_crystal_sys)
            FOXS = F[0]['structure'].composition.oxi_state_guesses()
            Fox_states = FOXS[0]
            for fspecies in F[0]['structure'].species:
                f_str = str(fspecies)
                if Fox_states[f_str] <= 0:
                    FRE = f_str
                    break

            SOXS = S[0]['structure'].composition.oxi_state_guesses()
            Sox_states = SOXS[0]
            for sspecies in S[0]['structure'].species:
                s_str = str(sspecies)
                if Sox_states[s_str] <= 0:
                    SRE = s_str
                    break
            
            gocp = Get_Oriented_Crystal_Plane()
            [f_OUC, f_vecs_cc, f_RRA, f_DS, FZT, FMV] = gocp.GOMUC(
                Film, f_crystal_sys, f_miller, F_vecs_cc, FRE)
            [s_OUC, s_vecs_cc, s_RRA, s_DS, SZT, SMV] = gocp.GOMUC(
                Substrate, s_crystal_sys, s_miller, S_vecs_cc, SRE)
            [final_SVCC, SDA, final_FVCC, FDA] = self.Get_Rotated_Match_Vectors(
                s_vecs_cc, f_vecs_cc)
            #print('final svcc final fvcc',final_SVCC,final_FVCC)

            sub_pos = self.Find_000_in_Match_Lattice(s_OUC)
            film_pos = self.Find_000_in_Match_Lattice(f_OUC)
            max_x = math.ceil(np.abs(final_SVCC[0][0])+np.abs(
                final_SVCC[1][0]))+(np.abs(
                    s_OUC[sub_pos][1].atomic_radius))
            max_y = math.ceil(np.abs(final_SVCC[0][1])+np.abs(
                final_SVCC[1][1]))+(np.abs(
                    s_OUC[sub_pos][1].atomic_radius))
            if self.ang == False:
                if -1 < self.shl < 1:
                    self.shl = np.ceil(np.abs(self.shl))
                if -1 < self.shw < 1:
                    self.shw = np.ceil(np.abs(self.shw))
                Scaled_max_x = max_x*np.abs(self.shl)
                Scaled_max_y = max_y*np.abs(self.shw)
            else:
                if self.shl < max_x:
                    Scaled_max_x = max_x
                else:
                    Scaled_max_x = self.shl
                if self.shw < max_y:
                    Scaled_max_y = max_y
                else:
                    Scaled_max_y = self.shw

            s_is_film = False
            SMOL = self.Make_Oriented_Matching_Lattice(
                s_OUC, s_vecs_cc, s_RRA, Scaled_max_x,
                Scaled_max_y, SDA, SZT, SMV, s_is_film)
            f_is_film = True
            FMOL = self.Make_Oriented_Matching_Lattice(
                f_OUC, f_vecs_cc, f_RRA, Scaled_max_x,
                Scaled_max_y, FDA, FZT, FMV, f_is_film)
            
            x_strain = (final_SVCC[0][0]-final_FVCC[0][0])/final_FVCC[0][0]
            y_strain = (final_SVCC[1][1]-final_FVCC[1][1])/final_FVCC[1][1]
            
            z_disp = (np.abs(s_OUC[sub_pos][1].atomic_radius))+(
            np.abs(f_OUC[film_pos][1].atomic_radius))+0.5

            final_film_lattice = self.Cut_and_Strain_Lattice(
                FMOL, z_disp, x_strain, y_strain, Scaled_max_x,
                Scaled_max_y, f_is_film)
            final_sub_lattice = self.Cut_and_Strain_Lattice(
                SMOL, z_disp, x_strain, y_strain, Scaled_max_x,
                Scaled_max_y, s_is_film)

            if self.PI == True: # make plot area larger
                [sx, sy] = self.Make_Python_Image_Sub(final_sub_lattice)
                [fx, fy] = self.Make_Python_Image_Film(final_film_lattice,
                                                       z_disp)
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
            
                plt.plot(sx,sy, 'ko', markersize=18)
                plt.plot(fx,fy, 'ro', markersize=9)
                plt.xlim(-1*Scaled_max_x, Scaled_max_x)
                plt.ylim(-1*Scaled_max_y, Scaled_max_y)

                ax.set_aspect('equal', adjustable='box')
                plot_name = "{}_on_{}_{}.png".format(f_form, s_form, num)
                plt.savefig(plot_name)

            final_film_lattice.extend(final_sub_lattice)
            num_of_atoms = str(len(final_film_lattice))
            comment_string = [
                f_form, 'on', s_form, 'with', np.around(x_strain,3),
                'film strain along the x-axis and', np.around(y_strain,3),
                'film strain along the y-axis.']
            joined_comment_string = ' '.join(map(str, comment_string))
            NoA = [num_of_atoms, '', '', '']
            JCS = [joined_comment_string, '', '', '']
            final_df = []
            final_df.append(NoA)
            final_df.append(JCS)
            for atomWC in final_film_lattice:
                entry = [atomWC[0], atomWC[1][0], atomWC[1][1], atomWC[1][2]]
                final_df.append(entry)

            if self.sf == True:
                sf = "ST"
            else:
                sf = "SF"
            FDF = pd.DataFrame(final_df)
            FDF.to_csv(r'c:pandas.txt', header=None, index=None, sep=' ')
            xyz = XYZ.from_file(r'c:\pandas.txt')
            xyz_file_name = "{}_{}{}{}_on_{}_{}{}{}_{}_{}.xyz".format(
                f_form, f_miller[0], f_miller[1], f_miller[2], s_form,
                s_miller[0], s_miller[1], s_miller[2], num, sf)
            xyz.write_file(xyz_file_name)
            os.remove(r'c:\pandas.txt')
    
def Vec_Angle_Rad(vec1, vec2): # changed here rounding ok?
    '''
    Gets the angle between two vectors. Is always positive.
    '''
    VecDot = np.dot(vec1,vec2)
    magx1 = fast_norm(vec1)
    magx2 = fast_norm(vec2)
    #print('vd mx1 mx2',VecDot,magx1,magx2)
    #print('inside arccos',VecDot/(magx1*magx2))
    vec_angle = np.arccos(np.around(VecDot/(magx1*magx2),8))
    return vec_angle
    
