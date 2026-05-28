# Solar-Powered Irrigation Spatial Optimisation

This repository contains the code developed for my MSc thesis project on the spatial suitability and techno-economic optimisation of solar-powered irrigation systems.

The project assesses whether solar-powered groundwater irrigation can provide a cost-effective water supply configuration for irrigated agriculture, with an initial focus on Egypt. The workflow is designed to be adaptable to other countries by changing the input datasets and scenario configuration.

## Project Overview

The model combines crop water demand estimation, groundwater and river abstraction energy requirements, spatial data processing, and energy-system optimisation.

The core objective is to identify where different irrigation energy configurations are most suitable:

* Solar-powered pumping
* Diesel-powered pumping
* Grid-powered pumping
* Hybrid or scenario-specific configurations

The final outputs include spatial maps, cost indicators, sensitivity results, and thesis-ready figures.

## Repository Structure

```text
project/data/
```

Contains the input and processed datasets. Each data category follows a bronze / silver / gold structure:

* `bronze_raw/`: original source files, kept unchanged.
* `silver_processed/`: cleaned and harmonised datasets.
* `gold_derived/`: final model-ready variables used in the optimisation.

Main data categories:

* `crops/`: crop distribution, harvested area, and crop-specific water demand inputs.
* `climate/`: solar resource, temperature, precipitation, and climate scenario data.
* `groundwater/`: aquifer depth, transmissivity, abstraction constraints, and groundwater availability indicators.

```text
project/src/
```

Contains the main modelling code:

* `water_demand.py`: estimates spatial crop water demand.
* `river_abstraction_energy.py`: calculates energy requirements for river-based irrigation.
* `groundwater_abstraction_energy.py`: calculates pumping energy requirements for groundwater abstraction.
* `clover_interface.py`: connects the irrigation model to CLOVER.
* `energy_optimisation.py`: compares solar, diesel, and grid-powered configurations.
* `sensitivity_analysis.py`: tests alternative climate, cost, and policy assumptions.

```text
project/qgis/
```

Contains QGIS project files, layer styles, and exported maps used for spatial visualisation.

```text
project/outputs/
```

Contains selected thesis outputs, including figures, maps, tables, and scenario results.

```text
project/notebooks/
```

Contains exploratory notebooks used to inspect, validate, and document each major dataset and modelling step.

## Relationship with CLOVER

This project builds on CLOVER, an open-source energy-system modelling framework.

CLOVER is included as an external Git submodule under:

```text
external/CLOVER/
```

CLOVER is not my original work and remains under its original license. This repository contains my thesis-specific modelling work around CLOVER, including spatial preprocessing, irrigation water demand estimation, abstraction energy calculations, scenario design, optimisation logic, and post-processing.

## Modelling Pipeline

The workflow follows these steps:

1. Process crop, climate, and groundwater datasets.
2. Estimate crop water demand across spatial units.
3. Calculate energy requirements for river and groundwater abstraction.
4. Connect irrigation demand profiles to CLOVER.
5. Optimise the choice between solar, diesel, and grid-powered supply.
6. Run sensitivity analyses on climate, cost, and policy assumptions.
7. Produce maps, figures, and results for the thesis.

## Current Status

This repository is under active development as part of an MSc thesis. Some modules, data inputs, and scenario assumptions may change as the research develops.

Large raw datasets and intermediate geospatial files are excluded from GitHub. The repository focuses on the code, methodology, configuration structure, and selected outputs.
