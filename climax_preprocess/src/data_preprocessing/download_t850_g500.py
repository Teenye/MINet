import climate_learn as cl

root_directory = "/data1/tengyue/data/weather-air/nc/era5_2.8125"
cl.data.download_weatherbench(
    f"{root_directory}/2m_temperature",
    dataset="era5",
    variable="2m_temperature",
    resolution=2.8125  # optional, default is 5.625
)
cl.data.download_weatherbench(
    f"{root_directory}/geopotential_500",
    dataset="era5",
    variable="geopotential_500",
    resolution=2.8125  # optional, default is 5.625
)
cl.data.download_weatherbench(
    f"{root_directory}/temperature_850",
    dataset="era5",
    variable="temperature_850",
    resolution=2.8125  # optional, default is 5.625
)

# import climate_learn as cl

# # root_directory = "/data1/tengyue/data/mpi_5.625"
# # variable = "tas"
# # cl.data.download_mpi_esm1_2_hr(
# #     dst=f"{root_directory}/{variable}",
# #     variable=variable,
# #     years=(1975, 2015), # optional, (1850, 2015) is the default range
# # )

# root_directory = "/data1/tengyue/data/mpi_5.625"
# variable = "uas"
# cl.data.download_mpi_esm1_2_hr(
#     dst=f"{root_directory}/{variable}",
#     variable=variable,
#     years=(1975, 2015), # optional, (1850, 2015) is the default range
# )
# root_directory = "/data1/tengyue/data/mpi_5.625"
# variable = "vas"
# cl.data.download_mpi_esm1_2_hr(
#     dst=f"{root_directory}/{variable}",
#     variable=variable,
#     years=(1975, 2015), # optional, (1850, 2015) is the default range
# )

# # root_directory = "/data1/tengyue/data/mpi_5.625"
# # variable = "zg"
# # cl.data.download_mpi_esm1_2_hr(
# #     dst=f"{root_directory}/{variable}",
# #     variable=variable,
# #     years=(1979, 2015), # optional, (1850, 2015) is the default range
# # )