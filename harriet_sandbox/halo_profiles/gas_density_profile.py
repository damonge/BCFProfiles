
import numpy as np
import unyt
# import hdfstream
import swiftsimio as sw


### remote connection
# Connect to the hdfstream service and open the root directory
# root_dir = hdfstream.open("cosma", "/")

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



rvir_max = 3

# Open the z=0 snapshot from the L1_m10_DMO simulation and select this region  - edited to m9
snap_file = general_path + "snapshots/flamingo_0077/flamingo_0077.hdf5"


mass_bins = np.logspace(10.0, 15.0, 6)

for j in range(len(mass_bins) - 1):
    m_low = mass_bins[j]
    m_high = mass_bins[j+1]
    
    bin_mask = centrals & (halo_m200c >= m_low) & (halo_m200c < m_high)
    bin_indices = np.where(bin_mask)[0]

    if len(bin_indices) == 0:
        continue
        
    n_sample = min(50, len(bin_indices))
    selected_bin_indices = np.random.choice(bin_indices, size=n_sample, replace=False)


    densities_array = []
    r_mid_array = []


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
        print(f"gas_mas {gas_mass}")

        in_halo = (gas_halo_index == target_index)
        gas_pos = gas_pos[in_halo]
        gas_mass = gas_mass[in_halo]


        distances = np.linalg.norm(gas_pos - target_pos, axis=1)
        dist_values = distances.value if hasattr(distances, "value") else distances
        r_max_val = r_max.value if hasattr(r_max, "value") else r_max

        mass_array, r_edges = np.histogram(
            distances, 
            range=[0.0, r_max_val],
            bins=10,
            weights=gas_mass
        )

        print(f"mass array {mass_array}")

        r_edges = unyt.unyt_array(r_edges, r_max.units)
        volume = 4*np.pi/3*(r_edges[1:]**3-r_edges[:-1]**3)
        density = mass_array / volume
        densities_array.append(density)
        print(densities_array)
    

    densities_array = unyt.unyt_array(densities_array)
    bin_average = np.mean(densities_array, axis=0)

    r_mid = (r_edges[1:] + r_edges[:-1]) * 0.5

    log_low = int(np.log10(m_low))
    log_high = int(np.log10(m_high))
    
    np.save(f'bin_average_mass_{log_low}_to_{log_high}_bounded.npy', bin_average)
    np.save(f'r_array_mass_{log_low}_to_{log_high}_bounded.npy', r_edges)