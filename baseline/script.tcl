open_component project_baseline

# set top function of the HLS design
set_top top_kernel

# add source file
add_files top.cpp

# add testbench
add_files -tb host.cpp

# FPGA part and clock configuration
# default frequency is 100 MHz
set_part {xczu3eg-sbva484-1-e}

# C synthesis for HLS design, generating RTL
csynth_design

# C/RTL co-simulation (provides the accurate cycle count for the report)
cosim_design

# RTL implementation (place and route); provides clock period and resources
export_design -format ip_catalog -flow impl

exit
