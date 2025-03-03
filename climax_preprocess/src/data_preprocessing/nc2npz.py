# Standard library
import glob
import os

# Third party
import numpy as np
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm

NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
    "vorticity": "vo",
    "potential_vorticity": "pv",
    "total_cloud_cover": "tcc",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "total_cloud_cover",
    "land_sea_mask",
    "orography",
    "lattitude",
]

PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
    "vorticity",
    "potential_vorticity",
]

VAR_TO_UNIT = {
    "2m_temperature": "K",
    "10m_u_component_of_wind": "m/s",
    "10m_v_component_of_wind": "m/s",
    "mean_sea_level_pressure": "Pa",
    "surface_pressure": "Pa",
    "toa_incident_solar_radiation": "J/m^2",
    "total_precipitation": "m",
    "total_cloud_cover": None,  # dimensionless
    "land_sea_mask": None,  # dimensionless
    "orography": None,  # dimensionless
    "geopotential": "m^2/s^2",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "temperature": "K",
    "relative_humidity": "%",
    "specific_humidity": "kg/kg",
    "voriticity": "1/s",
    "potential_vorticity": "K m^2 / (kg s)",
}

DEFAULT_PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925]

CONSTANTS = ["orography", "land_sea_mask", "slt", "lattitude", "longitude"]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}


HOURS_PER_YEAR = 8736  # 8760 --> 8736 which is dividable by 16


def nc2np(path, variables, years, save_dir, partition, num_shards_per_year):
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}

    constants_path = os.path.join(path, "constants.nc")
    constants_are_downloaded = os.path.isfile(constants_path)

    if constants_are_downloaded:
        constants = xr.open_mfdataset(
            constants_path, combine="by_coords", parallel=True
        )
        constant_fields = [VAR_TO_NAME[v] for v in CONSTANTS if v in VAR_TO_NAME.keys()]
        constant_values = {}
        for f in constant_fields:
            constant_values[f] = np.expand_dims(
                constants[NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)
            ).repeat(HOURS_PER_YEAR, axis=0)
            if partition == "train":
                normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
                normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

    for year in tqdm(years):
        np_vars = {}

        # constant variables
        if constants_are_downloaded:
            for f in constant_fields:
                np_vars[f] = constant_values[f]

        # non-constant fields
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc"))
            ds = xr.open_mfdataset(
                ps, combine="by_coords", parallel=True
            )  # dataset for a single variable
            code = NAME_TO_VAR[var]

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                # remove the last 24 hours if this year has 366 days
                if code == "tp":  # accumulate 6 hours and log transform
                    tp = ds[code].to_numpy()
                    tp_cum_6hrs = np.cumsum(tp, axis=0)
                    tp_cum_6hrs[6:] = tp_cum_6hrs[6:] - tp_cum_6hrs[:-6]
                    eps = 0.001
                    tp_cum_6hrs = np.log(eps + tp_cum_6hrs) - np.log(eps)
                    np_vars[var] = tp_cum_6hrs[-HOURS_PER_YEAR:]
                else:
                    np_vars[var] = ds[code].to_numpy()[-HOURS_PER_YEAR:]

                if partition == "train":
                    # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly]
                        normalize_std[var] = [var_std_yearly]
                    else:
                        normalize_mean[var].append(var_mean_yearly)
                        normalize_std[var].append(var_std_yearly)

                clim_yearly = np_vars[var].mean(axis=0)
                if var not in climatology:
                    climatology[var] = [clim_yearly]
                else:
                    climatology[var].append(clim_yearly)

            else:  # pressure-level variables
                assert len(ds[code].shape) == 4
                all_levels = ds["level"][:].to_numpy()
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = ds.sel(level=[level])
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[
                        -HOURS_PER_YEAR:
                    ]

                    if partition == "train":
                        # compute mean and std of each var in each year
                        var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                        if f"{var}_{level}" not in normalize_mean:
                            normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                            normalize_std[f"{var}_{level}"] = [var_std_yearly]
                        else:
                            normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                            normalize_std[f"{var}_{level}"].append(var_std_yearly)

                    clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
                    if f"{var}_{level}" not in climatology:
                        climatology[f"{var}_{level}"] = [clim_yearly]
                    else:
                        climatology[f"{var}_{level}"].append(clim_yearly)

        assert HOURS_PER_YEAR % num_shards_per_year == 0
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            if not constants_are_downloaded or var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            if not constants_are_downloaded or var not in constant_fields:
                mean, std = normalize_mean[var], normalize_std[var]
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (
                    (std**2).mean(axis=0)
                    + (mean**2).mean(axis=0)
                    - mean.mean(axis=0) ** 2
                )
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean.mean(axis=0)
                normalize_mean[var] = mean
                if var == "total_precipitation":
                    normalize_mean[var] = np.zeros_like(normalize_mean[var])
                normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )


def convert_nc2npz(
    root_dir,
    save_dir,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
):
    assert (
        start_val_year > start_train_year
        and start_test_year > start_val_year
        and end_year > start_test_year
    )
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)

    os.makedirs(save_dir, exist_ok=True)

    nc2np(root_dir, variables, train_years, save_dir, "train", num_shards)
    nc2np(root_dir, variables, val_years, save_dir, "val", num_shards)
    nc2np(root_dir, variables, test_years, save_dir, "test", num_shards)

    # save lat and lon data
    ps = glob.glob(os.path.join(root_dir, variables[0], f"*{train_years[0]}*.nc"))
    x = xr.open_mfdataset(ps[0], parallel=True)
    lat = np.array(x["lat"])
    lon = np.array(x["lon"])
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)


root_dir = "/public/home/jfhu/weather_data/climax/era5/1.40625deg_algined_1hour"
res_dir = "/public/home/jfhu/weather_data/climax/era5_1.40625deg_npz"
convert_nc2npz(root_dir=root_dir,save_dir=res_dir,variables=["2m_temperature","10m_u_component_of_wind","10m_v_component_of_wind"],start_train_year=1981,
        start_val_year=2016,
        start_test_year=2017,
        end_year=2019,num_shards=16)