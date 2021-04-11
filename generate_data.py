#%%
import pandas as pd
from matplotlib import pyplot as plt
from math import gcd
from functools import reduce
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import csv

#%%
########### DATA IMPORT ###############

data = pd.read_excel('DATA.xlsx', index_col=0,
                     sheet_name=None)

data['Hull Girder Stiffness Values'] = data['Hull Girder Stiffness Values'].drop(['Note '])
data['Segment mass distribution'] = data['Segment mass distribution'].drop(['Warning '])
data['Hull Girder Stiffness Values'] = data['Hull Girder Stiffness Values'].drop(['Note  This value is not the “real ship” neutral axis location. '])
data['Bonjean Data'] = data['Bonjean Data'].drop(['Notes ']).dropna()

for k, v in data.items():
    data[k].columns = data[k].columns.str.strip() # remove whitespace in column names
    data[k].rename(index = lambda x: x.strip() if isinstance(x, str) else x, inplace=True) # remove whitespace in indices
    print('#######################################################################################')
    print(k)
    print(v)
    print()
    print('#######################################################################################')

LOA = data['Principal Dimensions']['Prototype']['LOA (m)']
LBP = data['Principal Dimensions']['Prototype']['LBP (m)']

# %%
########### SET 1 ###############
sets = {}
sets['set1'] = 'NEW_ITTC_Hull' # filename and title. No spaces allowed


#%%
########### SET 2 ###############
set2_1 = '1 1 1 0 0 0' # input tags
set2_2 = '2 1 1 1 0' # output tags
sets['set2'] = set2_1 + ' ' + set2_2


#%%
########### SET 3 ###############

bulkheads = np.array([data['Bonjean Data']['Station position (m)'][1], *data['Location of Bulkheads']['Aft bhd (m)'][1:], data['Bonjean Data']['Station position (m)'][30]])

def find_best_dx(bulkheads = bulkheads):
    min_error = np.infty
    for i in np.arange(10, 401, step=2):
        dx = LOA/i
        
        rms = np.sqrt(np.mean(np.square((bulkheads - data['Segment Details']['Aft end'].to_numpy()[0] ) % dx)))
        max_error = max((bulkheads - data['Segment Details']['Aft end'].to_numpy()[0] ) % dx)
        plt.plot(i, max_error, 'x')
        
        if min_error > max_error:
            min_rms = rms
            min_error = max_error
            min_i = i
            min_dx = dx
    
    plt.grid(which='both', axis='both')
    plt.xlabel('No. sections')
    plt.ylabel('Max error from bulkheads (m)')
    plt.title('{} is the optimum no. sections. Max error : {:.3f} m'.format(min_i, min_error))
    plt.show()
    return min_dx, min_i, min_rms, min_error, ((bulkheads - data['Segment Details']['Aft end'].to_numpy()[0] ) % dx)

modes = 6
dx, num_sections, *_ = find_best_dx()

sets['set3'] = [num_sections, modes, dx]


# %%
########### SET 4 ###############

g = 9.80665
E = data['Hull Girder Material Properties']['Young Modulus']['E (steel)'] / 1000
poisson_ratio = 0.303
G = E/(2 * (1 + poisson_ratio))

sets['set4'] = [g, E, G]
#%%
########### SET 5 ###############
tol = 10**-14
wtol = 10**-6

I = float(data['Hull Girder Stiffness Values']['Prototype']['Midship section inertia'].split(' ')[0]) # second moment (m^4)
fam = 1 # added mass factor (1 for dry case)
disp = float(data['Principal Dimensions']['Prototype']['Displacement'] .split(' ')[0]) * 1000 # disp (kg)
wmin = (1.1 / (np.pi * (LOA**2))) * np.sqrt( (E/1000 * I * (10**5)) / (fam * disp / LOA) )
dw = 1.0
wnext = 1.01

sets['set5'] = [tol, wtol, wmin, dw, wnext]


#%%
########### SET 6 ###############
def create_trapezoid(area, centroid, h):
    a = ( 2 * area ) / ( h * ( 1 + ( ( (3 * centroid) - (2 * h)) / (h - (3 * centroid)) ) ) )
    b = a * ( 3 * centroid - 2 * h) / (h - 3 * centroid)
    m = (a - b) / h
    return m, b

def calc_kyy(index, min_x, max_x, section_function):
    segment_mass = data['Segment mass distribution'].at[index, 'Mass (t)']
    segment_centroid =  data['Segment mass distribution'].at[index, 'Local LCG (m)']
    
    k_yy_global = data['Segment mass distribution'].at[index, 'Kyy (m)']
    num_sections_in_segment = (data['Segment Details'].at[index, 'Segment length']) / (max_x - min_x)
    
    global_rotational_inertia = segment_mass * (k_yy_global**2 - segment_centroid)
    section_rotational_inertia = global_rotational_inertia / num_sections_in_segment
    
    p1 = section_function(min_x)
    p2 = section_function(max_x)
    section_centroid = (max_x - min_x) * (p2 + 2 * p1) / (3 * (p1 + p2))
    section_mass = np.trapz(y = [p1, p2], dx = (max_x - min_x) )
    k_yy = np.sqrt((section_rotational_inertia / section_mass) - section_centroid)
    return k_yy
    
#%%
def find_section_mass(min_x, max_x):
    segment_aft = data['Segment Details']['Aft end']
    segment_fwd = data['Segment Details']['Fwd end']
    
    aft_end_segment = [loc for loc in segment_aft if loc <= min_x][-1]
    try:
        fwd_end_segment = [loc for loc in segment_fwd if loc >= max_x][0]
    except IndexError:
        fwd_end_segment = segment_fwd.to_list()[-1]
    
    aft_segment_number = segment_aft.to_list().index(aft_end_segment)
    fwd_segment_number = segment_fwd.to_list().index(fwd_end_segment)
    
    if aft_segment_number == fwd_segment_number:
        index = data['Segment mass distribution'].index.values[aft_segment_number]
        h = data['Segment Details'].at[index, 'Segment length']
        segment_mass = data['Segment mass distribution'].at[index, 'Mass (t)']
        segment_centroid =  data['Segment mass distribution'].at[index, 'Local LCG (m)']
        m, y1 = create_trapezoid(segment_mass, segment_centroid, h)
        x1 = aft_end_segment
        f = lambda x : (m * (x - x1) + y1)
        p1 = f(min_x)
        p2 = f(max_x)
        dx = max_x - min_x
        
        section_mass = np.trapz(y = [p1, p2], dx = dx)
        # k_yy = data['Segment mass distribution'].at[index, 'Kyy (m)']
        k_yy = calc_kyy(index, min_x, max_x, f)
    
    else:
        bulkhead_loc = data['Segment Details'].at[data['Segment mass distribution'].index.values[aft_segment_number], 'Fwd end']
        
        aft_index = data['Segment mass distribution'].index.values[aft_segment_number]
        aft_segment_mass = data['Segment mass distribution'].at[aft_index, 'Mass (t)']
        aft_segment_centroid =  data['Segment mass distribution'].at[aft_index, 'Local LCG (m)']

        h = data['Segment Details'].at[aft_index, 'Segment length']
        dx1 = bulkhead_loc - min_x
        m, y1 = create_trapezoid(aft_segment_mass, aft_segment_centroid, h)
        x1 = aft_end_segment
        f = lambda x : (m * (x - x1) + y1)
        p1 = f(min_x)
        p2 = f(bulkhead_loc)
        aft_section_mass = np.trapz(y = [p1, p2], dx = dx1)
        
        # k_yy1 = data['Segment mass distribution'].at[aft_index, 'Kyy (m)']
        k_yy1 = calc_kyy(aft_index, min_x, bulkhead_loc, f)
        
        fwd_index = data['Segment mass distribution'].index.values[fwd_segment_number]
        fwd_segment_mass = data['Segment mass distribution'].at[fwd_index, 'Mass (t)']
        fwd_segment_centroid =  data['Segment mass distribution'].at[fwd_index, 'Local LCG (m)']
        
        h = data['Segment Details'].at[fwd_index, 'Segment length']
        dx2 = max_x - bulkhead_loc
        m, y1 = create_trapezoid(fwd_segment_mass, fwd_segment_centroid, h)
        x1 = aft_end_segment = [loc for loc in segment_aft if loc <= max_x][-1]
        f = lambda x : (m * (x - x1) + y1)
        p1 = f(bulkhead_loc)
        p2 = f(max_x)
        fwd_section_mass = np.trapz(y = [p1, p2], dx = dx2)
        
        # k_yy2 = data['Segment mass distribution'].at[fwd_index, 'Kyy (m)']
        k_yy2 = calc_kyy(fwd_index, bulkhead_loc, max_x, f)

        
        section_mass = aft_section_mass + fwd_section_mass
        k_yy = (k_yy1 + k_yy2)/2
    
    stations = data['Bonjean Data']['Station position (m)']
    keel_height = data['Bonjean Data']['Height of local underside of keel above baseline (m)']
    draft = data['Principal Dimensions'].at['Draft (m)', 'Prototype']
    local_drafts = draft - keel_height
    
    draft_spl = InterpolatedUnivariateSpline(stations, local_drafts, k=1)
    draft_prop = draft_spl((min_x + max_x)/2) / draft
    
    return section_mass, k_yy, draft_prop

def generate_section_masses(set3 = sets['set3'], data = data):
    num_sections, _, dx = set3
    section_masses = []
    vnert = []
    shear = []
    rnert = []
    kyys = []
    local_lcgs = []
    overall_lcg = data['Principal Dimensions']['Prototype']['LCG from AP (m)']
    
    midship_SMA = float(data['Hull Girder Stiffness Values']['Prototype']['Midship section inertia'].split(r' ')[0])
    midship_shear = float(data['Hull Girder Stiffness Values']['Prototype']['Midship section effective shear area'].split(r' ')[0])
    
    min_x = data['Segment Details']['Aft end'].to_numpy()[0]
    max_x = min_x + dx
    while len(section_masses) < num_sections:
        mass, kyy, draft_prop = find_section_mass(min_x, max_x)
        section_masses.append(mass)
        vnert.append(midship_SMA)
        shear.append(midship_shear)
        rnert.append(mass * kyy)
        kyys.append(kyy)
        local_lcgs.append((min_x + max_x) / 2)
        min_x = max_x
        max_x += dx

    Kyy = ( ( np.sum([ (section_masses[i] * (kyys[i]**2))
                      + (section_masses[i] * (local_lcgs[i]**2))
                      for i in range(len(section_masses)) ])
             - ((disp/1000) * overall_lcg ** 2) )
           / (disp/1000)
           ) ** 0.5
    
    print(Kyy)
    
    x = np.linspace(data['Segment Details']['Aft end'].to_numpy()[0], max_x, num_sections)
    plt.plot(x, section_masses, '-')
    plt.xlabel('Distance from AP (m)')
    plt.ylabel('Section mass (t)')
    plt.title('Error in total mass : {:.1f} kg'.format((np.sum(section_masses)*1000 - disp)))
    plt.grid(axis='both', which='both')
    plt.show()
    
    plt.plot(x, kyys, '-', label='Section')
    plt.plot(data['Segment mass distribution']['Local LCG (m)'] + data['Segment Details']['Aft end'], data['Segment mass distribution']['Kyy (m)'], 'x', label='Segment data points')
    plt.xlabel('Distance from AP (m)')
    plt.ylabel('Kyy (m)')
    plt.grid(axis='both', which='both')
    plt.legend()
    plt.show()
    return section_masses, vnert, shear, rnert, kyys, local_lcgs

smass, vnert, shear, rnert, kyys, local_lcgs = generate_section_masses()

sets['set6'] = [[smass[i], vnert[i], shear[i], rnert[i]] for i in range(num_sections)]


# %%
########### SET 10 ###############

sets['set10'] = [data['Principal Dimensions'].at[('Water density kg/m3'), ('Prototype')]/1000] # tonne/m3


# %%
########### SET 11 ###############
def generate_bonjean_data(set3 = sets['set3'], data = data):
    num_sections, *_ = set3
    
    stations = data['Bonjean Data']['Station position (m)']
    areas = 2 * data['Bonjean Data']['Derived Half Section Area (m2)']
    beams = 2 * data['Bonjean Data']['Derived Half Beam (m)']
    keel_height = data['Bonjean Data']['Height of local underside of keel above baseline (m)']
    drafts = data['Principal Dimensions'].at['Draft (m)', 'Prototype']
    local_drafts = drafts - keel_height

    plt.plot(stations, areas/10, 'x', label='Area data point')
    plt.plot(stations, beams, 'o', label='Beam data point')
    plt.plot(stations, local_drafts, '*', label= 'Draft data point')
    
    area_spl = InterpolatedUnivariateSpline(stations, areas, k=2)
    beam_spl = InterpolatedUnivariateSpline(stations, beams, k=2)
    draft_spl = InterpolatedUnivariateSpline(stations, local_drafts, k=2)
    
    xs = np.linspace(start = data['Segment Details']['Aft end'].to_numpy()[0], stop = data['Segment Details']['Fwd end'].to_numpy()[-1], num = num_sections+1)
    section_areas = area_spl(xs)
    section_beams = beam_spl(xs)
    section_drafts = draft_spl(xs)
    
    plt.plot(xs, section_areas/10, 'b-', label='Area spline')
    plt.plot(xs, section_beams, 'r-', label='Beam spline')
    plt.plot(xs, section_drafts, 'g-', label='Draft spline')
    plt.grid(axis='both', which='both')
    plt.xlabel('Distance from AP (m)')
    plt.ylabel('Area/10 (m2), Beam (m), Draft (m)')
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    set11 = np.empty( shape = (num_sections+1, 3) )
    i = 0
    for area, beam, draft in zip(section_areas, section_beams, section_drafts):
        if beam * draft < area:
            original_beam = beam
            original_draft = draft
            while beam * draft < area:
                if beam > draft:
                    draft += 0.01
                else:
                    beam += 0.01
            print('Sectional Area coefficient was > 1. Beam {:.4f} --> {:.4f}. Draft {:.4f} --> {:.4f}'.format(original_beam, beam, original_draft, draft))
        set11[i, :] = [area, beam, draft]
        i += 1
    
    return set11

sets['set11'] = generate_bonjean_data()


# %%
########### SET 14 ###############

NF = -100 # No. non-dimensional frequenciy parameters for 2-d hydrodynmaic propeties
NSPEED = 3 # No of ship's speeds.    1 <= NSPEED <= 10 
NHEAD = 1 # No of ship's headings  1 <= NHEAD <= 7 
NFREQ = -1 # No of wave frequencies  2 <= NFREQ <= 351. 
            # If NFREQ is -negative, then MARS will calculate NFREQ and the wave frequencies based on a specified stepsize,s.
            # In this case the absolute value of NFREQ is the number of frequency stepsizes listed in Set 18. 
WA = 1 # wave amplitude (m)
DAMFA = 2.0 # structural damping magnification factor. Applied structural damping = DAMFA x Kumai's (1958) value. 

sets['set14'] = [NF, NSPEED, NHEAD, NFREQ, WA, DAMFA]

#%%
########### SET 15 ###############
OMT_FLAG = 2 # 0 uses uniform spacing, 1 uses log10 spacing, 2 uses Ln spacing

sets['set15'] = [OMT_FLAG]


#%%
########### SET 16 ###############
speeds = np.array([0, 10, 20]) # speed in m/sec.
sets['set16'] = speeds * 0.5144


#%%
########### SET 17 ###############
headings = 180 # Heading in degrees. 180 is head seas

sets['set17'] = headings

#%%
########### SET 18 ###############

start_freq = 0.2
step_size = 0.04
end_freq = 6.0
sets['set18'] = [start_freq, step_size, end_freq]


#%%
########### SET 19 ###############

# only required if I6 == 1
XPRIME = 10
NTIME = 5
DTIME = 6
sets['set19'] = [XPRIME, NTIME, DTIME]

#%%
########### WRITE TO FILE ###############
if False:
    filename = sets['set1'] + '.inp'
    with open(filename, 'w') as f:
        for set_number in sets.keys():
            if len(list(np.shape(sets[set_number]))) == 0: # if data is a single number or string
                f.write(str(sets[set_number]))
                f.write('\n')
        
            elif len(list(np.shape(sets[set_number]))) == 1: # if data is a 1D list
                string = " ".join(str(v) for v in sets[set_number])
                f.write(string)
                f.write('\n')

            elif len(list(np.shape(sets[set_number]))) == 2: # if data is a 2D list
                for row in sets[set_number]:
                    string = " ".join('{:.10f}'.format(v) for v in row)
                    f.write(string)
                    f.write('\n')
            f.write('\n')
                
    print('########### DONE writing to {} ###########'.format(filename))
       
# %%
