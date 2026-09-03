
import numpy as np
import unyt
# import hdfstream
import swiftsimio as sw
# import matplotlib.pyplot as plt
from contexttimer import Timer

### remote connection 
# Connect to the hdfstream service and open the root directory
# root_dir = hdfstream.open("cosma", "/")

# # # Open the z=0 halo catalogue from the L1_m10_DMO simulation  - edited to m9
# general_path = "FLAMINGO/L1_m9/L1_m9/"
# soap_file = root_dir[general_path + "SOAP-HBT/halo_properties_0077.hdf5"]
# soap = sw.load(soap_file)

## local connection
general_path = '/mnt/extraspace/damonge/Datasets/Flamingo/FLAMINGO/L1_m9/L1_m9/'

# Open the z=0 halo catalogue from the L1_m10_DMO simulation  - edited to m9
soap_file = general_path + "SOAP-HBT/halo_properties_0077.hdf5"
soap = sw.load(soap_file)

# Get halo positions, masses and indexes
halo_pos = soap.input_halos.halo_centre
halo_m200c = soap.spherical_overdensity_200_crit.total_mass
halo_m200c = halo_m200c.to('Msun')
r200c = soap.spherical_overdensity_200_crit.soradius
r200c = r200c.to('Mpc')
halo_index = soap.input_halos.halo_catalogue_index
centrals = soap.input_halos.is_central == 1


# Print basic diagnostics on your halo masses
# log_m_min = 10.0
# log_m_max = 15.0
# valid_halos = centrals & (np.log10(halo_m200c) > log_m_min) & (np.log10(halo_m200c) < log_m_max)
# central_indices = np.where(valid_halos)[0]
# print(f"Number of valid central halos in mass range: {len(central_indices)}")
# print(f"Total halos in catalog: {len(halo_m200c)}")
# print(f"Total central halos: {np.sum(centrals)}")
# print(f"Minimum halo mass: {np.min(halo_m200c):.2e}")
# print(f"Maximum halo mass: {np.max(halo_m200c):.2e}")
# print(f"Log10 mass range: {np.min(np.log10(halo_m200c)):.2f} to {np.max(np.log10(halo_m200c)):.2f}")


rvir_max = 3

# Open the z=0 snapshot from the L1_m10_DMO simulation and select this region  - edited to m9
snap_file = general_path + "snapshots/flamingo_0077/flamingo_0077.hdf5"
# snap_file  = root_dir["FLAMINGO/L1_m9/L1_m9/snapshots/flamingo_0077/flamingo_0077.hdf5"]

mass_bins = np.logspace(10.0, 15.0, 6)

for j in range(len(mass_bins) - 1):
    print(f"Mass bin {j}")
    m_low = mass_bins[j]
    m_high = mass_bins[j+1]
    
    bin_mask = centrals & (halo_m200c >= m_low) & (halo_m200c < m_high)
    bin_indices = np.where(bin_mask)[0]

    if len(bin_indices) == 0:
        continue
        
    n_sample = min(50, len(bin_indices))
    selected_bin_indices = np.random.choice(bin_indices, size=n_sample, replace=False)

    halo_gas_pressure = []

    for idx in selected_bin_indices:
        target_pos = halo_pos[idx, :]
        target_index = halo_index[idx]
        r_max = r200c[idx] * rvir_max
        region = [[x - r_max, x + r_max] for x in target_pos]
        mask = sw.mask(snap_file)
        mask.constrain_spatial(region)
        snap = sw.load(snap_file, mask=mask)

        gas_pos = snap.gas.coordinates
        gas_halo_index = snap.gas.halo_catalogue_index
        gas_mass = snap.gas.masses
        gas_temperature = snap.gas.temperatures
        gas_ne = snap.gas.electron_number_densities
        gas_density = snap.gas.densities
        print(f"Mass before {len(gas_mass)}")
        # print(f"Gas temp before {len(gas_temperature)}")
        # print(f"Gas ne before {len(gas_ne)}")


        in_halo = (gas_halo_index == target_index)
        gas_pos = gas_pos[in_halo]
        gas_mass = gas_mass[in_halo]
        gas_density = gas_density[in_halo]
        gas_temperature = gas_temperature[in_halo]
        gas_ne = gas_ne[in_halo]
        print(f"Mass after {len(gas_mass)}")
        # print(f"Gas temp after {len(gas_temperature)}")
        # print(f"Gas ne after {len(gas_ne)}")
        

        particle_volume = gas_mass / gas_density
        distances = np.linalg.norm(gas_pos - target_pos, axis=1)
        dist_values = distances.value if hasattr(distances, "value") else distances
        r_max_val = r_max.value if hasattr(r_max, "value") else r_max
        
        weights=gas_ne*gas_temperature*particle_volume
        weights_vals = weights.value if hasattr(weights, "value") else weights
        pres_array, r_edges = np.histogram(
            dist_values, 
            range=[0.0, r_max_val],
            bins=10,
            weights=weights_vals
        )
        pres_array = unyt.unyt_array(pres_array, weights.units)
        r_edges = unyt.unyt_array(r_edges, r_max.units if hasattr(r_max, "units") else "Mpc")
        vol_bins = 4*np.pi/3*(r_edges[1:]**3-r_edges[:-1]**3)
        pressure_halo = pres_array / vol_bins
        halo_gas_pressure.append(pressure_halo)
        

    pressure_array = unyt.unyt_array(halo_gas_pressure)
    pressure_array = pressure_array.to("K/cm**3")
    kb = 8.617333262*1E-5
    pressure_array = pressure_array * kb   #in eV/cm**3

    bin_average = np.mean(pressure_array, axis=0)

    r_edges = np.linspace(0, rvir_max, 11)
    r_mid = (r_edges[1:]+r_edges[:-1])*0.5
    log_low = int(np.log10(m_low))
    log_high = int(np.log10(m_high))
    

    np.save(f'bound_bin_average_pressure_{log_low}_to_{log_high}.npy', bin_average)
    np.save(f'bound_r_array_pressure_{log_low}_to_{log_high}.npy', r_mid)