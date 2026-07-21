# How I used data from Busch et al 2024

18 July 2026, John F. Raffensperger, john.raffensperger@gmail.com, john.raffensperger.org.

My 2021 paper (corrected 2024) used forestry bids that I took from various places in the literature. I based those bids on only a few tree types and no land constraints. After seeing Busch et al. (2024), I decided to use their extensive dataset on forestry instead. Their dataset has something like 89 million "pixels" representing plots of land across the earth.

My repository here documents my export of the Busch et al dataset for SMDAMAGE integration. My goal is to simulate a land manager at each pixel, show SMDAMAGE prices to the manager, and invite the manager to offer a forestry contract into the SMDAMAGE auction. I used Claude to write much of this code. Nevertheless, I take responsibility for all errors in my code.

This repository is not the official Busch et al. distribution site. If you use my repository here, you should also cite the Busch et al (2024) paper and their public Zenodo data deposit. I acknowledge Busch et al for their tremendous work. They produced an excellent data set. My work is better as a result of theirs. I thank them for their effort.

## Busch et al code and data

My work here starts with the Busch et al. Zenodo deposit, especially:
- `08_input_dtas.zip` for the country-level input `.dta` files, under `Input/` in this repository.
- `12_stata_code.zip` for the original methodological scripts, under `DO code/` in this repository.
- `00_readme_Busch2024.docx` for upstream file descriptions

I treat the Stata scripts in `DO code/` as the source of logic for Busch et al. The main code is `DO code/1. Model loop all data.do`.Their .dta files are my main source of data; their code also contains important constants. I keep a locally corrected Stata file here: `DO code/1. Model loop all data corrected.do`. I include that corrected file because I identified a likely soil-carbon bug in the original `.do` file, described below in "Note on .DO file correction".

A note on terminology: Busch et al use the term "rotation year" and "harvest year" to mean a year in which a crop is harvested, typically resulting in carbon emissions rather than removal. They reasonably calculate costs over 30 years as though a land manager were planning over 30 years rather than the harvest period. Their "rotation year" is a bit confusing, since a plan for natural regeneration will not have a harvest. The SMDAMAGE auction depends on "contracts", where a land manager promises to manage a plot of land in a specific way for a period of time. I use "contract years" (and "contract length" synonymously). Based on careful analysis of $bid/(°C warming) and modest simplification, SMDAMAGE contracts could run as short as 20 years and as long as 120 years. If the SMDAMAGE contract does have a harvest, it is at the end of the contract, since the land manager could bid for another project of the same length. 

## Repository structure

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
- `JFR code/3_import_k_means_csv_to_sqlite.py` reads k-means CSV inputs from `Output/Kmeans_temp_files/` and writes `Pixel_bids` and per-contract carbon schedule tables to `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`. When run directly, creates four covering indexes on `Pixel_bids` for fast downstream queries.
- `JFR code/4_move_Busch_data_to_SMDAMAGE.py` reads `Pixel_bids` from `Output/Databases/Busch2024_to_SMDAMAGE.sqlite` and builds per-cluster bid curve tables (`cluster_forestry_bid_curves_N`) for each contract length and discount rate.

## Setup 

This repository has one Python dependency entrypoint: `requirements.txt`.
One-command setup option (recommended on Windows PowerShell): `./scripts/setup.ps1`

From the repository root on Windows PowerShell:
1. Create and activate a virtual environment:
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
2. Install dependencies:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
3. Run the pipeline scripts from the Quick Start section above.

## Work flow overview

From the repository root, run the core phase-one pipeline in this order:
1. Build/refresh intermediate per-contract CSV exports: `python "JFR code/1a_import_Busch2024_to_SMDAMAGE.py"`
2. Build/refresh k-means schedules and assignments: `python "JFR code/2_k_means_carbon_removal.py"`
3a. Load k-means CSV outputs into SQLite: call `run_loader()` from `JFR code/3_import_k_means_csv_to_sqlite.py`
3b. Create `Pixel_bids` covering indexes: `python "JFR code/3_import_k_means_csv_to_sqlite.py"`
4. Build forestry bid curves: `python "JFR code/4_move_Busch_data_to_SMDAMAGE.py"`
The code writes the primary data products (intermediate CSV files and SQLite databases) to `Output/Databases/` and `Output/Kmeans_temp_files/`.

Details below.

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
- computes carbon schedules for natural regeneration and plantation options based on a 3% discount rate,
- evaluates bids and bid scores across contract lengths 20, 30, ..., 120 years,
- chooses each pixel's best option and best contract length,
- write a CSV file for each contract length to `Output/Databases/undiscounted_contracts_XXX.csv`. I decided to write output to CSV files before SQLite import to keep the heavy numeric pass simple, memory-safe, and restartable. The code streams to CSV, which is lighter than writing to SQLite which needs indexing.

I had to choose a discount rate in selecting the contract length and I stick with that carbon schedule in tons C/year to avoid multiple contract lengths for a given pixel. Given the carbon schedule (calculated based on a 0.03 discount rate), I calculate bids with discount rates 0, 0.015, 0.3, and 0.6, discounting the opportunity cost/year and the harvest revenue back to the year 1 of the contract. This is not double discounting! Rather it approximates the carbon schedule with respect to discount rates different than 0.03.

### 2. Clustering of pixels, 2_k_means_carbon_removal.py

A Busch et al pixel is a plot of land. We can think of that plot as a bidder in SMDAMAGE. 1a_import_Busch2024_to_SMDAMAGE.py finds that bidder's best bid for SMDAMAGE. But 89 million bidders is too big for SMDAMAGE, so I use a k-means to cluster pixels based on carbon schedule into 100 groups, treating them as identical within the group, except for bid.

The code writes the solution to CSV files. (I will import into Busch2024_to_SMDAMAGE.sqlite with the cleverly named program 3_import_k_means_csv_to_sqlite.py, next). This code is fiddly to run. I finally got a satisfactory run that took about 18 hours to run. I think the limiting factor was 8GB RAM when I wish I had 64GB RAM, resulting in caching to the solid state drive. Claude was helpful in getting it to run faster and use less memory. Eventually I got Python to use the GPU.

2_k_means_carbon_removal.py:
	1. Reads `undiscounted_contracts_XXX.csv` by contract length,
	2. Runs a sweep-based k-allocation pass to estimate error as a function of cluster count,
	3. Chooses the cluster count for each contract length based on marginal error, with 100 total clusters across all contract lengths,
	4. Runs the final k-means pass for each contract length,
	5. Writes cluster centers and assignments to CSV files in `Output/Kmeans_temp_files/`

Summary results:
Contract
length	# pixels	hectares		# clusters
	20	10,399,106	999,775,298		16
	30	10,815,908	994,859,185		16
	40	5,256,921	483,240,557		10
	50	5,689,848	542,955,473		13
	60	2,505,180	234,482,214		7
	70	1,623,950	151,199,734		8
	80	1,121,736	103,727,989		5
	90	862,483		78,841,709		5
	100	809,700		73,235,760		5
	110	756,737		67,899,812		5
	120	18,463,262	1,578,573,429	16
Now we can think of a bidder as a land manager proposing a carbon schedule (one unique carbon schedule per cluster) and a specific contract length (e.g., 20 years), with total area of the cluster like 999,775,298/14 hectares. So we have 100 bidders (one per cluster) instead of 89 million (one per pixel). In SMDAMAGE, they will offer a bid in every year of the 200+ year auction schedule. I calculate the bid curve for each cluster, further subdividing the cluster bidder into a separate bidder for each step of the bid curve.

Later in the workflow, SMDAMAGE project file create_database.py reads Busch2024_to_SMDAMAGE.sqlite and loads forestry bidders, one per group, into the SMDAMAGE auction.

### 3. Import clustered outputs into SQLite, 3_import_k_means_csv_to_sqlite.py

`JFR code/3_import_k_means_csv_to_sqlite.py` imports the k-means CSV outputs into `Output/Databases/Busch2024_to_SMDAMAGE.sqlite`. It uses a direct insert-only pipeline (no temporary tables, no row-level checks) with bulk-load SQLite pragmas for maximum throughput.
- reads `undiscounted_contracts_XXX.csv` (per contract length) from `Output/Databases/`
- reads `k_means_carbon_removal_contract_years_*_centers.csv` to build per-cluster carbon schedule tables (`cluster_carbon_schedules_N`)
- streams each year's assignments CSV and undiscounted CSV together to insert `Pixel_bids` rows with `cluster_id` already populated (insert-only; no update pass)
- reads `k_means_carbon_removal_contract_years_*_overall.csv` for pixel-count metadata
- commits per contract year
- when run directly (`__main__`), creates four covering indexes on `Pixel_bids` — one per NPV discount rate column — structured as `(contract_years, cluster_id, NPV_col, pixel_id, area_ha)` for fast downstream bid-curve queries; this step is separate from loading so the slow load does not need to be repeated if only indexing is needed

### 4. Build bid curves, 4_move_Busch_data_to_SMDAMAGE.py

`JFR code/4_move_Busch_data_to_SMDAMAGE.py` reads `Pixel_bids` from `Busch2024_to_SMDAMAGE.sqlite` and builds per-cluster bid curves for each contract length and discount rate. It requires the covering indexes created by script 3.
- sets large read cache and memory-mapped I/O pragmas at start
- for each contract length and each of four discount rates (0%, 1.5%, 3%, 6%), computes bid steps in a single SQL window-function CTE that processes all clusters at once; pixels are sorted by NPV and bucketed into steps of approximately 3% of cluster area each, with the top 1% by cost omitted
- writes `cluster_forestry_bid_curves_N` tables (cluster_id, discount_rate_00, bid_step, npv_max_per_ha, step_area_ha) — typically ~33 steps per cluster per rate
- creates the secondary index on each curve table after all rows are inserted
- verifies bid-step monotonicity within each cluster and rate
- commits per contract year

## Note on .DO file correction

In my local review, I identified an error in the original `DO code/1. Model loop all data.do`. In its section labeled `Discounted present value of soil carbon, w/harvest`, the file contains: `replace nr_soilC=nr_soilC + (1-d)^(`i'-1) * nrsoil`.

I interpret that line as reusing the natural-regeneration soil-carbon increment `nrsoil` inside the with-harvest loop, where plantation soil-carbon logic was intended. That would use the wrong soil-carbon parameter in the harvest branch. I raised this `.do`-file issue in correspondence with the Busch paper authors, who graciously responded. 

In my corrected file, I change that line to: `replace nr_soilC=nr_soilC + (1-d)^(`i'-1) * plantsoil`. I preserve that local correction in: `DO code/1. Model loop all data corrected.do`.

# Data and publications

Raffensperger, John F. (2021, 2024) A price on warming with a supply chain directed market. Discover Sustainability 2, 2. https://doi.org/10.1007/s43621-021-00011-4

Busch, J., Bukoski, J. J., Cook-Patton, S. C., Griscom, B., Kaczan, D., Potts, M. D., Yi, Y., and Vincent, J. R. (2024). Cost-effectiveness of natural forest regeneration and plantations for climate mitigation. Nature Climate Change, 14(9), 996-1002. https://doi.org/10.1038/s41558-024-02068-1.

Busch, J., Bukoski, J. J., Cook-Patton, S. C., Griscom, B., Kaczan, D., Potts, M. D., Yi, Y., and Vincent, J. R. (2024). Data for "Cost-effectiveness of natural forest regeneration and plantations for climate mitigation". Zenodo. https://doi.org/10.5281/zenodo.11372275. Zenodo record title: Data for "Cost-effectiveness of natural forest regeneration and plantations for climate mitigation".

Their Zenodo record states that all data associated with the paper are publicly available there. Their record includes `00_readme_Busch2024.docx`, `08_input_dtas.zip`, `09_output_dtas.zip`, `10_sensitivity_dtas.zip`, and `12_stata_code.zip`. Their Zenodo dataset is published under CC BY 4.0. It shows a later version record, but I organize this repository around the local working files present here rather than trying to track every upstream revision automatically.