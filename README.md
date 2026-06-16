# Busch2024 Data Documentation, John F. Raffensperger, 2026-06-16.

The SMDAMAGE model previously used forestry bids that I took from various places in the literature. Those bids are based on only a few tree types and no land constraints. After seeing Busch et al. (2024), I decided to use their extensive dataset on forestry instead. Their dataset has explicit land constraints and a wide variety of tree types.

This repository documents my use of the Busch et al dataset. I use this repository for two main purposes:
- to preserve the methodological lineage from the original Busch et al. (2024) Stata workflow,
- to document my local export, validation, correction, and schedule-compression work for SMDAMAGE integration.

This repository is not the official Busch et al. distribution site. Their publication and data depository are below.

## Publication

Busch, J., Bukoski, J. J., Cook-Patton, S. C., Griscom, B., Kaczan, D., Potts, M. D., Yi, Y., and Vincent, J. R. (2024). Cost-effectiveness of natural forest regeneration and plantations for climate mitigation. Nature Climate Change, 14(9), 996-1002. https://doi.org/10.1038/s41558-024-02068-1.

# Data

Busch, J., Bukoski, J. J., Cook-Patton, S. C., Griscom, B., Kaczan, D., Potts, M. D., Yi, Y., and Vincent, J. R. (2024). Data for "Cost-effectiveness of natural forest regeneration and plantations for climate mitigation". Zenodo. https://doi.org/10.5281/zenodo.11372275. Zenodo record title: Data for "Cost-effectiveness of natural forest regeneration and plantations for climate mitigation".

Their Zenodo record states that all data associated with the paper are publicly available there. Their record includes `00_readme_Busch2024.docx`, `08_input_dtas.zip`, `09_output_dtas.zip`, `10_sensitivity_dtas.zip`, and `12_stata_code.zip`. Their Zenodo dataset is published under CC BY 4.0. It shows a later version record, but I organize this repository around the local working files present here rather than trying to track every upstream revision automatically.

## Repository scope

I organize the top-level folders as follows:
- `DO code/`: original and locally corrected Stata scripts.
- `Input/`: country-level Busch `.dta` input files.
- `Output/maps/`: original Busch `maps_*.dta` files
- `JFR code/`: local Python utilities for export, inspection, clustering, and SQLite import.
- `Output/Databases/Busch2024_dta_outputs.sqlite`: Collected data from Busch's dta files.
- `Output/Kmeans_temp_files/`: k-means centers, assignments, summary, and overall CSV files
- `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`: processed Busch data ready for export to SMDAMAGE.

Current scripts follow this structure:
- `JFR code/1_k_means_carbon_removal.py` writes k-means CSV outputs to `Output/Kmeans_temp_files/`
- `JFR code/2_import_k_means_csv_to_sqlite.py` reads k-means CSV inputs from `Output/Kmeans_temp_files/` and writes to `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`
- `JFR code/4_move_Busch_data_to_SMDAMAGE.py` reads from `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`; its current `__main__` block is a placeholder (`pass`)
- `JFR code/Old/draw_outputs_graph.py` writes `graph_of_variable_names.svg` to `Output/40_reports/`

## Quick Start

From the repository root, run the core phase-one pipeline in this order:
1. Build/refresh the base export database: `python "JFR code/0_import_Busch2024_to_SMDAMAGE.py"`
2. Build/refresh k-means schedules and assignments: `python "JFR code/1_k_means_carbon_removal.py"`
3. Import k-means CSV outputs back into SQLite: `python "JFR code/2_import_k_means_csv_to_sqlite.py"`
4. (Optional, currently no-op) Run helper code scaffold: `python "JFR code/4_move_Busch_data_to_SMDAMAGE.py"`

The code writes the primary data products to `Output/Databases/` and `Output/Kmeans_temp_files/`.

## Reproducible Setup Entrypoint

This repository now has a single Python dependency entrypoint: `requirements.txt`.
One-command setup option (recommended on Windows PowerShell): `./scripts/setup.ps1`

From the repository root on Windows PowerShell:
1. Create and activate a virtual environment:
	- `python -m venv .venv`
	- `.\.venv\Scripts\Activate.ps1`
2. Install dependencies:
	- `python -m pip install --upgrade pip`
	- `python -m pip install -r requirements.txt`
3. Run the pipeline scripts from the Quick Start section above.

## Source-of-truth and local conventions

For methodology questions, I treat the Stata scripts in `DO code/` as the source of truth. The main lineage anchor is: `DO code/1. Model loop all data.do`

The main local Python export lineage anchor is: `JFR code/0_import_Busch2024_to_SMDAMAGE.py`

I also keep a locally corrected Stata file here: `DO code/1. Model loop all data corrected.do`. I include that corrected file because I identified a likely soil-carbon bug in the original `.do` file, described below.

## How I created `Busch2024_to_SMDAMAGE.sqlite`

This section documents the local workflow that I used to create the SMDAMAGE-style SQLite database.

### 1. Obtain the upstream Busch data

I start from the Busch et al. Zenodo deposit. My local workflow depends especially on:
- `08_input_dtas.zip` for the country-level input `.dta` files
- `12_stata_code.zip` for the original methodological scripts
- `00_readme_Busch2024.docx` for upstream file descriptions

In this repository, I place the country `.dta` files under `Input/`.

### 2. Use the Busch methodology as the reference model

Busch et al. (2024) compute, for each pixel:
- admissibility after biome and data-quality filtering
- natural-regeneration and plantation carbon accumulation
- root-to-shoot adjustments and soil-carbon terms
- establishment and opportunity costs
- harvest-related wood-product storage and revenue terms
- branch-level cost-effectiveness for natural regeneration versus plantation
- the selected lower-cost option for each pixel

In the original Stata workflow, Busch et al. (2024) set a 30-year horizon and a 5% discount rate. In my local SMDAMAGE export pipeline, I preserve the same overall economic logic but use a 35-year export horizon in the Python implementation.

### 3. Run my local Python export pipeline

The main local export script is: `JFR code/0_import_Busch2024_to_SMDAMAGE.py`

That script is a local downstream export layer. Claude helped me write parts of this Python code during my repository work, but the script follows the Busch et al. methodological structure rather than replacing it.

In that script, I:
- read country `.dta` files with `pandas.read_stata`
- filter out excluded biomes and invalid rows
- convert `area_m2` to `area_ha`
- inflate the Busch cost inputs from 2011 USD to 2020 USD
- map plantation `type` codes to a working genus
- assign continent and ecological-zone root-to-shoot ratios
- compute annual carbon schedules for natural regeneration and plantation options
- compute plantation harvest years using a Faustmann-style rule and then bucket them to a smaller set of rotation years for downstream tractability
- compare cost-effectiveness and select the lower-cost pixel-level option
- write one row per pixel to `Undiscounted_dta_output`
- write one row per pixel-year to `tC_per_h_per_year`

The export script initializes the database with two main tables:
- `Undiscounted_dta_output`
- `tC_per_h_per_year`

Key pixel-level export fields include:

- `country`
- `pixel_id`
- `plantation_genus`
- `selected_option`
- `selected_A`
- `selected_k`
- `selected_rotation_year`
- `area_ha`
- `crop_va_USD_per_ha_per_year`
- `selected_establishment_cost_USD_per_ha`
- `p_USD_per_tC_harvested`

### 4. Net present value and discounting details

Busch et al. (2024) combine biological growth with discounted economic terms. In my local export, I keep that structure but store the final carbon schedules in undiscounted annual form for downstream use in SMDAMAGE.

Important local constants in my Python export are:

- `TIME_HORIZON_YEARS = 35`
- `DISCOUNT_RATE = 0.05`

I use the discount rate in two main ways:

- to choose harvest years using a Faustmann-style decision rule
- to carry forward the Busch logic for recurring plantation establishment costs and discounted comparison logic

The key plantation-with-harvest cost-effectiveness expression in my local export is:

`pl_whrv_costeff_USD_per_tCO2 = (crop_va * Y + pl_npv_estcost_USD_per_ha - p * pl_harvested_tC_per_ha) / (3.67 * pl_whrv_total_tC_per_ha)`

I interpret that expression as follows:

- `crop_va * Y` is the per-hectare opportunity-cost term over the modeled horizon
- `pl_npv_estcost_USD_per_ha` is the plantation establishment-cost term produced by the Busch logic
- `p * pl_harvested_tC_per_ha` is the harvest-revenue offset
- `3.67` converts tons of carbon to tons of CO2

For natural regeneration, I use the analogous cost expression with `nr_cost` instead of the plantation establishment-cost term and without a harvest-revenue offset.

Important note:
- the harvest revenue term is not a bare subtraction of `p`
- it is `p` multiplied by harvested carbon per hectare

### 5. Local run mode used in this repository

In this repository, I configure the export script to run all available country files from `Input/` and to write directly to `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`.

The relevant script settings are:
- `RUN_ALL_PIXELS = True`
- `SKIP_SQLITE_EXPORT = False`
- inline settings that iterate all `.dta` files in `Input/`

### 6. Post-export clustering work

After I created the SMDAMAGE-style SQLite database, I added a schedule-compression workflow to reduce the number of forestry schedules.

Relevant scripts:
- `JFR code/3_choose_harvest_years.py`
- `JFR code/1_k_means_carbon_removal.py`
- `JFR code/2_import_k_means_csv_to_sqlite.py`

I used JFR code/1_k_means_carbon_removal.py to compress pixel-level carbon-removal schedules into forestry bidders for SMDAMAGE.
The k-means algorithm clustered pixels within each selected_rotation_year on year-by-year tC_per_ha_per_year profiles.

I then used JFR code/2_import_k_means_csv_to_sqlite.py to load those cluster centers and assignments into Output/Databases/Busch2024_to_SMDAMAGE.sqlite.
The database stores the cluster centers in table carbon_removal_schedules and each pixel receives a cluster_index.

Next, I used JFR code/4_move_Busch_data_to_SMDAMAGE.py to export bidder metadata (forestry_bidder_metadata.csv)
and bidder-specific bid-step curves (forestry_bid_steps_long.csv) from per-pixel costs and areas.

In that export step, each (rotation_year, cluster_index) pair is one bidder, summed area becomes available_area_mhectares,
and bid steps are built by sorting per-pixel bid_cost_per_ha and accumulating area.

This keeps SMDAMAGE tractable while preserving heterogeneity in carbon timing, land availability, and costs.

As of 2026-04-18, this clustered schedule import produced 71 forestry schedules distributed across rotation years 6, 10, 17, and 35,
and I used it to write schedule rows into `carbon_removal_schedules` and assign `cluster_index` values in `Undiscounted_dta_output`.

## Summary of the original `.do`-file issue
In my local review, I identified one likely methodological error in the original `DO code/1. Model loop all data.do`.

In the section labeled `Discounted present value of soil carbon, w/harvest`, the original file contains:
`replace nr_soilC=nr_soilC + (1-d)^(`i'-1) * nrsoil`

In my corrected file, I change that line to:
`replace nr_soilC=nr_soilC + (1-d)^(`i'-1) * plantsoil`

I interpret the original line as reusing the natural-regeneration soil-carbon increment `nrsoil` inside the with-harvest loop, where plantation soil-carbon logic was intended. That would bias the comparison between natural regeneration and plantation cases by using the wrong soil-carbon parameter in the harvest branch.

I preserve that local correction in: `DO code/1. Model loop all data corrected.do`

## Note on correspondence with the authors

I raised this `.do`-file issue in correspondence with the Busch paper authors. This repository does not include private emails or a verbatim correspondence archive, so I limit the public summary to the following:
- I identified a likely soil-carbon issue in the original Stata code
- the issue concerns the use of `nrsoil` versus `plantsoil` in the with-harvest soil-carbon loop
- I preserve a locally corrected `.do` file reflecting that change

If I later add a fuller correspondence record to the repository, I should update this section to cite that record directly.

## Plan for bringing the data into SMDAMAGE

This section summarizes my current working state in this repository as of phase one.

Current status from my latest workflow notes:
1. DONE: create the Busch SQLite export with pixel-level economics and annual carbon schedules.
2. DONE: compress pixel-level carbon-removal schedules with k-means and import cluster centers/assignments back into SQLite.
3. DONE: build bidder-specific forestry bid curves by `(selected_rotation_year, cluster_index)` from per-pixel costs and areas.
4. NEXT: move to the SMDAMAGE repository and complete ingestion/refactor there.

The key integration direction remains to replace the legacy fixed forestry layout in SMDAMAGE with bidder-specific long-format forestry inputs. Practical SMDAMAGE-side targets are:
- one forestry bidder per `(selected_rotation_year, cluster_index)` pair
- explicit bidder metadata (rotation year, contract years, available area)
- bidder-specific bid-step rows instead of a single shared wide grid
- bidder-specific sequestration schedules
- bidder-specific land-cap constraints in the optimization model

In short, my intended SMDAMAGE integration path is:
1. use the Busch SQLite export as the per-pixel source table
2. use clustered schedules from `carbon_removal_schedules`
3. construct long-format forestry bidder metadata, bid-step, and sequestration tables
4. refactor SMDAMAGE input loading to support those long-format forestry tables
5. replace the aggregate forestry land constraint with bidder-specific land caps

## Reproducibility notes

I use this repository as a working research repository rather than as a polished upstream software package. Reproducibility depends on keeping several lineage distinctions clear.

- `Busch2024_dta_outputs.sqlite` and `Busch2024_to_SMDAMAGE.sqlite` are different databases with different purposes
- the Stata `.do` files remain the methodological anchor for Busch et al. (2024)
- my Python export and k-means scripts are downstream transformation layers
- Claude wrote or helped write some of my local Python utilities, but I used those utilities to implement and inspect the Busch et al. logic rather than to redefine it
- my local corrections and downstream integration steps need explicit documentation because they are not part of the original Busch et al. release by default

## Recommended citation practice

If you use this repository for methodology or downstream integration, I recommend citing both the Busch et al. (2024) paper and the public Zenodo data deposit:

- Paper: https://doi.org/10.1038/s41558-024-02068-1
- Data: https://doi.org/10.5281/zenodo.11372275

If you rely on my local corrections or my downstream SMDAMAGE integration work in this repository, cite the relevant repository snapshot separately in your own publication workflow.

JFR 16 June 2026, john.raffensperger@gmail.com, john.raffensperger.org.
