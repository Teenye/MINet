import os

import click
import numpy as np
import xarray as xr
from tqdm import tqdm
from skimage.transform import resize


def get_mpi_str(year):
    
    if year < 1990:
        str = '197001010600-199001010000'
    elif year >= 1990 and year < 2010:
        str = '199001010600-201001010000'
    elif year >= 2010 and year < 2015:
        str = '201001010600-201501010000'
    else:
        str = None
            
    return str



## --path_era5 /public/CMIP/data/era_1.406/
## --path_mpi /public/home/jfhu/dataset_mpi/mpi/5.625deg/
@click.command()
@click.option("--path_mpi", type=str, default='/data1/tengyue/data/mpi_5.625/files')
@click.option("--path_era5", type=str, default='/data1/tengyue/data/era5_1.40625')
@click.option("--new_path_mpi", type=str, default='/data1/tengyue/data/mpi/5.625deg_algined')
@click.option("--new_path_era5", type=str, default='/data1/tengyue/data/era5/1.40625deg_algined')
@click.option("--mpi_res", type=str, default='5.625')
@click.option("--era5_res", type=str, default='1.40625deg')
@click.option(
    "--variables",
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'geopotential',
        'temperature'
    ],
)
def main(
    path_mpi,
    path_era5,
    new_path_mpi,
    new_path_era5,
    mpi_res,
    era5_res,
    variables
):
    era5_years = list(range(1979, 2019))

    for var in variables:
        print ('Aligning and regrid', var)
        os.makedirs(os.path.join(new_path_mpi, var), exist_ok=True)
        os.makedirs(os.path.join(new_path_era5, var), exist_ok=True)
        
        if var == "2m_temperature":
            mpi_var = 'tas_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn'
            mpi_name = 'tas'
            era5_name = 't2m'
            
        elif var == "10m_u_component_of_wind":
            mpi_var = 'uas_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn'
            mpi_name = 'uas'
            era5_name = 'u10'
            
        elif var == "10m_v_component_of_wind":
            mpi_var = 'vas_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn'
            mpi_name = 'vas'
            era5_name = 'v10'
            
        elif var == "geopotential":
            mpi_var = 'zg_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn'
            mpi_name = 'zg'
            era5_name = 'z'
            
        elif var == "temperature":
            mpi_var = 'ta_6hrPlevPt_MPI-ESM1-2-LR_historical_r1i1p1f1_gn'
            mpi_name = 'ta'
            era5_name = 't'
        
        
        for year in tqdm(era5_years):
            mpi_str = get_mpi_str(year)
            if mpi_str is not None:
                mpi_ds = xr.open_dataset(os.path.join(path_mpi, f"{mpi_var}_{mpi_str}.nc"))
                # 没有子目录所以不用var
                # era5_ds = xr.open_dataset(os.path.join(path_era5, var ,f"{var}_{year}_{era5_res}.nc"))
                era5_ds = xr.open_dataset(os.path.join(path_era5,var, f"{var}_{year}_{era5_res}.nc"))

                time_ids = np.isin(mpi_ds.time, era5_ds.time)
                mpi_ds = mpi_ds.sel(time=time_ids)
                
                
                # new_shape = (32, 64)
                
                # data = mpi_ds[mpi_name].values
                
                # if len(data.shape) == 4:
                #     reshaped_data = data.reshape(-1, data.shape[-2], data.shape[-1])
                #     resized_data = np.array([resize(frame, new_shape, anti_aliasing=True) for frame in reshaped_data])
                #     data = resized_data.reshape(data.shape[0], data.shape[1], new_shape[0], new_shape[1])
                # else:
                #     resized_data = np.array([resize(frame, new_shape, anti_aliasing=True) for frame in reshaped_data])
                #     data = resized_data.reshape(data.shape[0], new_shape[0], new_shape[1])
                
                # mpi_ds[mpi_name].values = data
                
                time_ids = np.isin(era5_ds.time, mpi_ds.time)
                era5_ds = era5_ds.sel(time=time_ids)
                mpi_ds.to_netcdf(os.path.join(new_path_mpi, var, f"{var}_{year}_{mpi_res}.nc"))
                era5_ds.to_netcdf(os.path.join(new_path_era5, var, f"{var}_{year}_{era5_res}.nc"))


if __name__ == "__main__":
    main()