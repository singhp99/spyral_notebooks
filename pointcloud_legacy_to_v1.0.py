import h5py 
import numpy as np
import tqdm
from pathlib import Path



def change_pc_legacy(run_num):
    
    pointcloud_legacy = f"/Volumes/researchEXT/O16/no_efield/PointcloudLegacy/run_00{run_num}.h5" if run_num < 100 else f"/Volumes/researchEXT/O16/no_efield/PointcloudLegacy/run_0{run_num}.h5"
    spyral_v1 = f"/Volumes/researchEXT/O16/no_efield/Pointcloud/run_00{run_num}.h5" if run_num < 100 else f"/Volumes/researchEXT/O16/no_efield/Pointcloud/run_0{run_num}.h5"
    file_exists = Path(pointcloud_legacy)
    
    if file_exists.exists():
        legacy_file = h5py.File(pointcloud_legacy, "r")
        legacy_ls = list(legacy_file.keys())[0]
        legacy_data = legacy_file[legacy_ls]
        min_event = legacy_data.attrs["min_event"]
        max_event = legacy_data.attrs["max_event"]
        
        with h5py.File(spyral_v1, "w") as f:
            new_group = f.create_group("cloud")
            new_group.attrs["min_event"] = min_event
            new_group.attrs["max_event"] = max_event
            
            for i, key in enumerate(legacy_data):
                data_lg = legacy_data[key][:]
                
                new_data = new_group.create_dataset(key, data=data_lg)
                
                new_data.attrs["orig_run"] =  run_num 
                new_data.attrs["orig_event"] = int(key.strip("cloud_")) 
                
                new_data.attrs["ic_amplitude"] = float(-1)
                new_data.attrs["ic_integral"] = float(-1)
                new_data.attrs["ic_centroid"] = float(-1)
                new_data.attrs["ic_multiplicity"] = float(-1)
                new_data.attrs["ic_sca_centroid"] = float(-1)
                new_data.attrs["ic_sca_multiplicity"] = float(-1)

    
def main():
    for run_num in tqdm.tqdm(range(54,170),desc="Filtering runs"):
        change_pc_legacy(run_num)
    
    
if __name__ == "__main__":
    main()
    