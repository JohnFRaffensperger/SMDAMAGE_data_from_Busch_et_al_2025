# Busch2024 Data Documentation, John F. Raffensperger, 2026-06-16.

My 2021 paper (corrected 2024) used forestry bids that I took from various places in the literature. I based those bids on only a few tree types and no land constraints. After seeing Busch et al. (2024), I decided to use their extensive dataset on forestry instead. Their dataset has something like 58 million "pixels" representing actual plots of land across the earth. This repository is not the official Busch et al. distribution site. Their publication and data depository are below.

My repository here documents my export of the Busch et al dataset for SMDAMAGE integration. My goal is to simulate a land manager at each pixel, show SMDAMAGE prices to the manager, and invite the manager to offer a forestry contract into the SMDAMAGE auction.

If you use this repository for methodology or downstream integration, I recommend citing both the Busch et al. (2024) paper and the public Zenodo data deposit.

I acknowledge Busch et al for their tremendous work. They produced an excellent data set. My work is better as a result of theirs. I thank them for their effort.

I used Claude to write much of this code. Nevertheless, I take responsibility for all errors in my code.

## Publications

Raffensperger, John F. (2021, 2024) A price on warming with a supply chain directed market. Discover Sustainability 2, 2. https://doi.org/10.1007/s43621-021-00011-4

Busch, J., Bukoski, J. J., Cook-Patton, S. C., Griscom, B., Kaczan, D., Potts, M. D., Yi, Y., and Vincent, J. R. (2024). Cost-effectiveness of natural forest regeneration and plantations for climate mitigation. Nature Climate Change, 14(9), 996-1002. https://doi.org/10.1038/s41558-024-02068-1. Zenodo data deposit: https://doi.org/10.5281/zenodo.11372275

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
- `Output/Databases/`: intermediate per-contract CSV files from `1a_import_Busch2024_to_SMDAMAGE.py`
- `Output/Kmeans_temp_files/`: k-means centers, assignments, summary, and overall CSV files
- `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`: processed Busch data ready for export to SMDAMAGE.

Current scripts follow this structure:
- `JFR code/1a_import_Busch2024_to_SMDAMAGE.py` reads Busch `.dta` inputs and writes one intermediate CSV per contract length to `Output/Databases/`
- `JFR code/2_k_means_carbon_removal.py` reads those intermediate CSV files and writes k-means outputs to `Output/Kmeans_temp_files/`
- `JFR code/3_import_k_means_csv_to_sqlite.py` reads k-means CSV inputs from `Output/Kmeans_temp_files/` and writes to `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`
- `JFR code/4_move_Busch_data_to_SMDAMAGE.py` reads from `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`; its current `__main__` block is a placeholder (`pass`)
- `JFR code/Old/draw_outputs_graph.py` writes `graph_of_variable_names.svg` to `Output/40_reports/`

## Quick Start

To be honest, please don't be hoping for a quick start. From the repository root, run the core phase-one pipeline in this order:
1. Build/refresh intermediate per-contract CSV exports: `python "JFR code/1a_import_Busch2024_to_SMDAMAGE.py"`
2. Build/refresh k-means schedules and assignments: `python "JFR code/2_k_means_carbon_removal.py"`
3. Import k-means CSV outputs into SQLite: `python "JFR code/3_import_k_means_csv_to_sqlite.py"`
4. (Optional, currently no-op) Run helper code scaffold: `python "JFR code/4_move_Busch_data_to_SMDAMAGE.py"`

The code writes the primary data products to `Output/Databases/` and `Output/Kmeans_temp_files/`.

## Reproducible Setup Entrypoint

This repository has one Python dependency entrypoint: `requirements.txt`.
One-command setup option (recommended on Windows PowerShell): `./scripts/setup.ps1`

From the repository root on Windows PowerShell:
1. Create and activate a virtual environment:
	- `python -m venv .venv`
	- `.\.venv\Scripts\Activate.ps1`
2. Install dependencies:
	- `python -m pip install --upgrade pip`
	- `python -m pip install -r requirements.txt`
3. Run the pipeline scripts from the Quick Start section above.

## Busch et al code and data

I start from the Busch et al. Zenodo deposit. My local workflow depends especially on:
- `08_input_dtas.zip` for the country-level input `.dta` files, under `Input/` in this repository.
- `12_stata_code.zip` for the original methodological scripts, under `DO code/` in this repository.
- `00_readme_Busch2024.docx` for upstream file descriptions

I treat the Stata scripts in `DO code/` as the source of logic for Busch et al. The main code is `DO code/1. Model loop all data.do`.Their .dta files are my main source of data; their code also contains important constants. I keep a locally corrected Stata file here: `DO code/1. Model loop all data corrected.do`. I include that corrected file because I identified a likely soil-carbon bug in the original `.do` file, described below in "Note on .DO file correction".

 ## How I created `Busch2024_to_SMDAMAGE.sqlite`

This section documents my workflow to create the SMDAMAGE SQLite database.

### 1. Calculate carbon schedules and bids, 1a_import_Busch2024_to_SMDAMAGE.py

The first script is: `JFR code/1a_import_Busch2024_to_SMDAMAGE.py`. For each pixel and forestry option, this code calculates the carbon schedule and associated bid. It then chooses the best growing option and contract length based on $bid/(°C warming).

For each pixel, Busch et al calculate the following:
- admissibility after biome and data-quality filtering
- natural-regeneration and plantation carbon accumulation
- root-to-shoot adjustments and soil-carbon terms
- establishment and opportunity costs
- harvest-related wood-product storage and revenue terms
- branch-level cost-effectiveness for natural regeneration versus plantation
- the selected lower-cost option for each pixel

In the original Stata workflow, Busch et al set a 30-year horizon and a 5% discount rate. Their main output is the $/ton C and then the resulting marginal abatement cost curves for forestry.

But SMDAMAGE does not price by $/ton C. It prices on warming at the dealine of 2125. I therefore do a different pricing calculation that accounts for the warming (actually cooling) of the proposed planting plan. Using 1a_import_Busch2024_to_SMDAMAGE.py on each pixel and growing option, I calculate the carbon removal schedule similar to how Busch et al did it; I found the derivative of their cumulative carbon function. Then I choose a contract length based on minimizing $bid/(°C warming), taking care with present values, etc. This is how a land manager would decide how to bid into the SMDAMAGE auction. This pricing mechanism is itself an important contribution.

1a_import_Busch2024_to_SMDAMAGE.py:
- reads country `.dta` files,
- applies Busch et al data cleaning and biome filters,
- computes carbon-removal schedules for natural regeneration and plantation options,
- evaluates bids and bid scores across contract lengths 20, 30, ..., 120 years,
- chooses each pixel's best option and best contract length,
- write a CSV file for each contract length to `Output/Databases/undiscounted_contracts_XXX.csv`. I decided to write output to CSV files before SQLite import to keep the heavy numeric pass simple, memory-safe, and restartable. The code streams to CSV, which is lighter than writing to SQLite which needs indexing.

### 2. Clustering of pixels, 2_k_means_carbon_removal.py

A Busch et al pixel is a plot of land. We can think of that plot as a bidder in SMDAMAGE. 1a_import_Busch2024_to_SMDAMAGE.py finds that bidder's best bid for SMDAMAGE. But 58 million bidders is too big for SMDAMAGE, so I use a k-means to cluster pixels into 100 groups,
based on carbon schedule, treating them as identical within the group, except for bid.

The code writes the solution to CSV files. (I will import into Busch2024_to_SMDAMAGE.sqlite with the cleverly named program 3_import_k_means_csv_to_sqlite.py, next). This code can run out of memory. It is fiddly to run. I finally got a satisfactory run 
that took about 18 hours to run. I think the limiting factor was 8GB RAM when I wish I had 64GB RAM, 
resulting in caching to the solid state drive. Claude was helpful in getting it to run faster and use less memory.

2_k_means_carbon_removal.py:
- reads `undiscounted_contracts_XXX.csv` by contract length,
- runs a sweep-based k-allocation pass to estimate error as a function of cluster count,
- chooses the cluster count for each contract length based on marginal error, with 100 total clusters across all contract lengths,
- runs the final k-means pass for each contract length,
- writes cluster centers and assignments to CSV files in `Output/Kmeans_temp_files/`

Later in the workflow, SMDAMAGE project file create_database.py reads Busch2024_to_SMDAMAGE.sqlite
and loads forestry bidders, one per group, into the SMDAMAGE auction.

### 3. Import clustered outputs into SQLite, 3_import_k_means_csv_to_sqlite.py

`JFR code/3_import_k_means_csv_to_sqlite.py` imports the k-means CSV outputs into `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`, populates clustered schedule tables, and writes per-pixel cluster assignments for downstream SMDAMAGE integration.

## Note on .DO file correction

In my local review, I identified one likely methodological error in the original `DO code/1. Model loop all data.do`. In their section labeled `Discounted present value of soil carbon, w/harvest`, the original file contains: `replace nr_soilC=nr_soilC + (1-d)^(`i'-1) * nrsoil`.

I interpret the original line as reusing the natural-regeneration soil-carbon increment `nrsoil` inside the with-harvest loop, where plantation soil-carbon logic was intended. That would bias the comparison between natural regeneration and plantation cases by using the wrong soil-carbon parameter in the harvest branch. I raised this `.do`-file issue in correspondence with the Busch paper authors, and they graciously responded. 

In my corrected file, I change that line to: `replace nr_soilC=nr_soilC + (1-d)^(`i'-1) * plantsoil`. I preserve that local correction in: `DO code/1. Model loop all data corrected.do`.

JFR 16 June 2026, john.raffensperger@gmail.com, john.raffensperger.org.
