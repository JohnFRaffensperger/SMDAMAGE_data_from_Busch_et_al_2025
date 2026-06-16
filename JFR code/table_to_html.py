# table_to_html.py | JFR Created 2026-03-23. Use this handy utility to run SQLite queries and view the results in HTML. It also exports CSVs.

import csv
from ctypes import *
from pathlib import Path
import sys
import time
import pandas, pandas.io.sql, sqlite3
import sqlite3

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Output"
if str(OUTPUT_DIR) not in sys.path: sys.path.insert(0, str(OUTPUT_DIR))
import LocalHTML

DEFAULT_DB_FILE = OUTPUT_DIR / "Busch2024_dta_outputs.sqlite"
DEFAULT_HTML_FILE = OUTPUT_DIR / "mytable.html"
DEFAULT_QUERY_CSV = OUTPUT_DIR / "myQueryOutput.csv"
DEFAULT_AGGREGATE_CSV = OUTPUT_DIR / "myAggregateOutput.csv"
DEFAULT_EXPORT_CSV = OUTPUT_DIR / "yourquery.csv"
PRIMARY_TABLE_NAME = "Busch2024_dta_outputs"

def getTableNameFromQueryString(query): # Tries to get the table name from the query string.
	start = query.find(' from ')
	substring = query[start+6:]
	end = min(substring.find('\t'), substring.find(' '))
	return query[start+6:start+6+end]

def makeHTML(rows, fieldIndices = None):
	outputfilename = DEFAULT_HTML_FILE
	if not rows:
		print ("No rows")
		return

	if fieldIndices:
		fieldnames = ['']*len(rows[0])
		for field in fieldIndices: fieldnames[fieldIndices[field]] = field
		with open(outputfilename, 'w') as f: f.write('<html>' + LocalHTML.table(rows, header_row = fieldnames) + '</html>')
	else:
		with open(outputfilename, 'w') as f: f.write('<html>' + LocalHTML.table(rows) + '</html>')
	print ("Wrote %i rows to %s." % (len(rows), outputfilename))
	return

def dumpQueryToCSV(query, databaseFileName = str(DEFAULT_DB_FILE)):
	con = sqlite3.connect(databaseFileName)
	table = pandas.io.sql.read_sql(query, con)
	table.to_csv(DEFAULT_EXPORT_CSV, index=False)
	con.close()
#dumpQueryToCSV("select * from mymaster limit 10", "eda.db")

def saveQueryToCSV (query, optionalFileName = str(DEFAULT_QUERY_CSV)):
	con = sqlite3.connect(str(DEFAULT_DB_FILE))
	cursor = con.cursor()
	cursor.execute(query)

	with open(optionalFileName, "w", newline='') as csv_file:
		csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
		csv_writer.writerow([i[0] for i in cursor.description]) # write headers
		csv_writer.writerows(cursor)
	con.commit()
	con.close()

# saveQueryToCSV ("select sum(area) as t_area, crop_npv_oppcost from maps_atg_schema group by crop_npv_oppcost")
# saveQueryToCSV ("select cast(crop_npv_oppcost as integer) as crop_npv_oppcost_int, sum(area) as t_area from maps_atg_schema where crop_npv_oppcost < 2000 group by cast(crop_npv_oppcost as integer)")
# saveQueryToCSV ("select cast(crop_npv_oppcost as integer) as crop_npv_oppcost_int, sum(area) as t_area from maps_atg_schema group by cast(crop_npv_oppcost as integer)")
# saveQueryToCSV ("select sum(area)	 as t_area from maps_atg_schema", 'totalarea.txt')

def show(tablename): tableToHtml("select * from %s limit 500" % tablename)

def resolve_query_table_name(query, databaseFileName = str(DEFAULT_DB_FILE)):
	if "maps_atg_schema" not in query: return query
	return query.replace("maps_atg_schema", PRIMARY_TABLE_NAME)

def tableToHtml(query, optionalFileName = str(DEFAULT_HTML_FILE), talk=True, databaseFileName = str(DEFAULT_DB_FILE), transpose = False):
	''' This function does the query, and writes the result to optionalFileName.
		It returns a dictionary of the field names and their indices in the rows,
		and the rows themselves.
		Optionally, it can look up some field name descriptions and field values.
	'''
	# 1. Run the requested query.
	if not Path(databaseFileName).exists():
		print("Sorry, database file '%s' does not exist." % databaseFileName) # Don't create a zero size database if it doesn't exit.
		return
	query = resolve_query_table_name(query, databaseFileName)
	# uri=True enables SQLite URI syntax; mode=ro opens read-only (ro), alternatives: rw (read-write, no create), rwc (read-write-create, default), memory (in-memory).
	con = sqlite3.connect(f"file:{databaseFileName}?mode=ro", uri=True) # Don't allow writing to the database.
	mycursor = con.cursor()
	execute_start = time.perf_counter()
	try: rowset = mycursor.execute(query)
	except Exception as e:
		print("Sorry, query '%s' failed, error %s." % (query, str(e)))
		con.close()
		return
	execute_elapsed = time.perf_counter() - execute_start
	if talk: print ("Finished execute in %.2f s, now fetching rows." % execute_elapsed)

	if rowset:
		fetch_start = time.perf_counter()
		fieldnames = [description[0] for description in rowset.description]
		fieldNameDict = {description[0]: desc_index for (desc_index, description) in enumerate(rowset.description)}
		rowList = list(rowset)
		counter = len(rowList)
		fetch_elapsed = time.perf_counter() - fetch_start
		if talk: print ("Finished fetch in %.2f s, now making HTML." % fetch_elapsed)
	con.close()

	# 4. Write the HTML file, transposing if requested.
	tablename = getTableNameFromQueryString(query)
#	print("tablename = ", tablename)
	write_start = time.perf_counter()
	with open(optionalFileName, 'w', encoding='utf-8') as htmlfile:
		if transpose:
			rowList.insert(0, fieldnames) # Put header at top.
			rowList = list(map(list, zip(*rowList))) # Transpose.
			rowList = sorted(rowList, key=lambda x:x[0].upper()) # sort by field name.
			htmlfile.write('<html><p>'+query+', ' + str(counter) + ' rows. </p>' + LocalHTML.table(rowList) + "</html>")
		else: htmlfile.write('<html><title>' + tablename + '</title><body><p>'+query+', ' + str(counter) + ' rows. </p>' + str(LocalHTML.Table(rowList, header_row = fieldnames)) + "</body></html>")
	write_elapsed = time.perf_counter() - write_start
	if talk: print ("Finished HTML write in %.2f s." % write_elapsed)
	print ("Found " + str(counter) + " rows to %s." % optionalFileName)
	return fieldNameDict, rowList

def printQueryRows(query, databaseFileName = str(DEFAULT_DB_FILE)): print((result[1] if (result := tableToHtml(query, optionalFileName='NUL', talk=False, databaseFileName=databaseFileName)) else []))

def aggregate(bucket_size = 1):
	'''Reads myQueryOutput.csv, aggregates t_area by crop_npv_oppcost rounded
	to multiples of (N*bucket_size). Writes to myAggregateOutput.csv.
	'''
	if not isinstance(bucket_size, int) or bucket_size <= 0: raise ValueError("bucket_size must be a positive integer.")
	input_file = DEFAULT_QUERY_CSV
	output_file = DEFAULT_AGGREGATE_CSV
	aggregated = {}

	with open(input_file, 'r', newline='', encoding='utf-8') as in_file:
		reader = csv.DictReader(in_file)
		for row in reader:
			if not row.get('t_area') or not row.get('crop_npv_oppcost'): continue
			rounded_cost = bucket_size*int(round(float(row['crop_npv_oppcost']) / float(bucket_size)))
			# 't_area' is square meters. Divide by 10000 to get hectares.
			aggregated[rounded_cost] = aggregated.get(rounded_cost, 0.0) + float(row['t_area'])/10000
			if rounded_cost > 1500: break

	with open(output_file, 'w', newline='', encoding='utf-8') as out_file:
		writer = csv.writer(out_file, quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(['crop_npv_oppcost_bucket', 't_area'])
		for bucket in sorted(aggregated.keys()): writer.writerow([bucket, aggregated[bucket]])

# aggregate(15)

# ---------------------------
# Examine Busch2024_dta_outputs.sqlite.
# Shows over 5 billion hectares available in the Busch2024 outputs.
# tableToHtml(f"select sum(area) as t_area from Busch2024_dta_outputs", databaseFileName = OUTPUT_DIR /"Busch2024_dta_outputs.sqlite")

# Curve of establishment cost and area.
# tableToHtml("select 10 * cast(round(selected_establishment_cost_USD_per_ha / 10.0, 0) as integer) as selected_establishment_cost_USD_per_ha_bucket_10, count(*) as row_count, sum(area_ha) as total_area_ha from Undiscounted_dta_output where selected_establishment_cost_USD_per_ha is not null group by 10 * cast(round(selected_establishment_cost_USD_per_ha / 10.0, 0) as integer) order by 10 * cast(round(selected_establishment_cost_USD_per_ha / 10.0, 0) as integer)", databaseFileName = OUTPUT_DIR / "Busch2024_to_SMDAMAGE.sqlite")

# 5 billion hectares, only 39 million missing mi_whrv_costeff.
# tableToHtml("""select 50 * cast(round(mi_whrv_costeff / 50.0, 0) as integer) as mi_whrv_costeff_bucket_50,
# 	count(*) as row_count, sum(area) as total_area from Busch2024_dta_outputs group by mi_whrv_costeff_bucket_50""",
# 	databaseFileName = OUTPUT_DIR / 'Busch2024_dta_outputs.sqlite')

# ---------------------------
# Examine Busch2024_dta_outputs.sqlite.
# Total area lost 39,078,488 ha.
# tableToHtml("select count(*) as mi_whrv_costeff_null, sum(area) as t_area from Busch2024_dta_outputs where mi_whrv_costeff is null or mi_whrv_costeff != mi_whrv_costeff limit 10000", databaseFileName = str(OUTPUT_DIR / 'Busch2024_dta_outputs.sqlite'))
# Only 28,779 missing fao_ecoz. Upstream field: nr_wohrv_total_tC_per_ha. Verified inputs: nr_A, nr_k, fao_ecoz, type. Applies the same row exclusions as export_Busch2024_to_SMDAMAGE.py: keep non-biome-13/14 rows and require nr_A and type to be non-missing.
# tableToHtml("""with filtered as (select * from Busch2024_inputs where (biomes is null or biomes != 13) and (biomes is null or biomes != 14) and nr_A is not null and nr_A = nr_A and type is not null and type = type),
# 	prepared as (select *, case when type in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) then 0 else 1 end as type_unmapped from filtered)
# 	select count(*) as row_count, sum(case when nr_A is null or nr_A != nr_A then 1 else 0 end) as nr_A_missing,
# 		sum(case when nr_k is null or nr_k != nr_k then 1 else 0 end) as nr_k_missing, sum(case when fao_ecoz is null or fao_ecoz != fao_ecoz then 1 else 0 end) as fao_ecoz_missing,
# 		sum(case when type is null or type != type then 1 else 0 end) as type_missing, sum(type_unmapped) as type_unmapped from prepared""", databaseFileName = str(OUTPUT_DIR / "Busch2024_inputs.sqlite"))

# Only 28,779 missing fao_ecoz. Upstream field: pl_whrv_total_tC_per_ha. Verified inputs: type, fao_ecoz, selected genus A, selected genus k. Applies the same row exclusions as export_Busch2024_to_SMDAMAGE.py.
# tableToHtml("""with filtered as (select * from Busch2024_inputs where (biomes is null or biomes != 13) and (biomes is null or biomes != 14) and nr_A is not null and nr_A = nr_A and type is not null and type = type), prepared as (select *,
# 	case when type = 1 then neev_A when type = 2 then brev_A when type = 3 then brde_A when type = 4 then neev_A when type = 5 then cunn_A when type = 6 then euca_A when type = 7 then nede_A when type = 8 then neev_A when type = 9 then pinu_A when type = 10 then brde_A when type = 11 then neev_A when type = 12 then brde_A when type = 13 then brev_A when type = 14 then brde_A when type = 15 then neev_A end as selected_genus_A,
# 	case when type = 1 then neev_k when type = 2 then brev_k when type = 3 then brde_k when type = 4 then neev_k when type = 5 then cunn_k when type = 6 then euca_k when type = 7 then nede_k when type = 8 then neev_k when type = 9 then pinu_k when type = 10 then brde_k when type = 11 then neev_k when type = 12 then brde_k when type = 13 then brev_k when type = 14 then brde_k when type = 15 then neev_k end as selected_genus_k from filtered)
# 	select count(*) as row_count, sum(case when type is null or type != type then 1 else 0 end) as type_missing,
# 		sum(case when fao_ecoz is null or fao_ecoz != fao_ecoz then 1 else 0 end) as fao_ecoz_missing,
# 		sum(case when selected_genus_A is null or selected_genus_A != selected_genus_A then 1 else 0 end) as selected_genus_A_missing,
# 		sum(case when selected_genus_k is null or selected_genus_k != selected_genus_k then 1 else 0 end) as selected_genus_k_missing from prepared""", databaseFileName = str(OUTPUT_DIR / "Busch2024_inputs.sqlite"))

# 447,715 crop_va missing. Applies the same row exclusions as export_Busch2024_to_SMDAMAGE.py.
# tableToHtml("""select count(*) as row_count, sum(case when crop_va is null or crop_va != crop_va then 1 else 0 end) as crop_va_missing from Busch2024_inputs""",
# 		databaseFileName = OUTPUT_DIR / "Busch2024_inputs.sqlite")

# Only 2373 missing nr_cost. Upstream field: nr_cost. Verified input: nr_cost. Applies the same row exclusions as export_Busch2024_to_SMDAMAGE.py.
# tableToHtml("""select count(*) as row_count, sum(case when nr_cost is null or nr_cost != nr_cost then 1 else 0 end) as nr_cost_missing from Busch2024_inputs where (biomes is null or biomes != 13) and (biomes is null or biomes != 14) and nr_A is not null and nr_A = nr_A and type is not null and type = type""", databaseFileName = str(OUTPUT_DIR / "Busch2024_inputs.sqlite"))

# Only 2373 missing ep_cost and np_cost. Upstream field: pl_npv_estcost_USD_per_ha. Verified inputs: type, ep_cost, np_cost, source_iso for the CHN cunn exception. Applies the same row exclusions as export_Busch2024_to_SMDAMAGE.py.
# tableToHtml("""with filtered as (select * from Busch2024_inputs where (biomes is null or biomes != 13) and (biomes is null or biomes != 14) and nr_A is not null and nr_A = nr_A and type is not null and type = type), prepared as (select *,
# 	case when type = 5 and source_iso = 'CHN' then np_cost when type = 5 then ep_cost when type = 6 then ep_cost when type in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15) then case when ep_cost is null or ep_cost != ep_cost or np_cost is null or np_cost != np_cost then null else 0 end end as selected_cost_inputs_complete_flag from filtered)
# 	select count(*) as row_count, sum(case when type is null or type != type then 1 else 0 end) as type_missing,
# 		sum(case when ep_cost is null or ep_cost != ep_cost then 1 else 0 end) as ep_cost_missing, sum(case when np_cost is null or np_cost != np_cost then 1 else 0 end) as np_cost_missing,
# 		sum(case when selected_cost_inputs_complete_flag is null then 1 else 0 end) as selected_cost_inputs_missing from prepared""", databaseFileName = str(OUTPUT_DIR / "Busch2024_inputs.sqlite"))

# 1,492,528 "ISO unmapped for P". Upstream field: p. Verified input: source_iso only. Applies the same row exclusions as export_Busch2024_to_SMDAMAGE.py.
# tableToHtml("""select count(*) as row_count, sum(case when source_iso is null or trim(source_iso) = '' then 1 else 0 end) as source_iso_missing,
# 		sum(case when source_iso not in ('AFG', 'AGO', 'ARG', 'ARM', 'ATG', 'AZE', 'BDI', 'BEN', 'BFA', 'BGD', 'BLZ', 'BOL', 'BRA', 'BRB', 'BTN', 'BWA', 'CAF', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COK', 'COM', 'CPV', 'CRI', 'CUB', 'DJI', 'DMA', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESH', 'ETH', 'FJI', 'FSM', 'GAB', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRD', 'GTM', 'GUF', 'GUY', 'HND', 'HTI', 'IDN', 'IND', 'IRN', 'IRQ', 'JAM', 'JOR', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KNA', 'LAO', 'LBN', 'LBR', 'LCA', 'LKA', 'LSO', 'MAR', 'MDG', 'MDV', 'MEX', 'MHL', 'MLI', 'MMR', 'MNG', 'MOZ', 'MRT', 'MUS', 'MWI', 'NAM', 'NER', 'NGA', 'NIC', 'NPL', 'NRU', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PLW', 'PNG', 'PRK', 'PRY', 'PSE', 'RWA', 'SDN', 'SEN', 'SLB', 'SLE', 'SLV', 'SOM', 'SSD', 'STP', 'SUR', 'SWZ', 'SYC', 'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKL', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TZA', 'UGA', 'URY', 'UZB', 'VCT', 'VEN', 'VNM', 'VUT', 'WSM', 'YEM', 'ZAF', 'ZMB', 'ZWE') then 1 else 0 end) as source_iso_unmapped_for_p from Busch2024_inputs where (biomes is null or biomes != 13) and (biomes is null or biomes != 14) and nr_A is not null and nr_A = nr_A and type is not null and type = type""", databaseFileName = str(OUTPUT_DIR / "Busch2024_inputs.sqlite"))

# ALL ZERO. Upstream field: pl_harvested_tC_per_ha. Verified inputs: type, selected genus A, selected genus k. Applies the same row exclusions as export_Busch2024_to_SMDAMAGE.py.
# tableToHtml("""with filtered as (select * from Busch2024_inputs where (biomes is null or biomes != 13) and (biomes is null or biomes != 14) and nr_A is not null and nr_A = nr_A and type is not null and type = type), prepared as (select *,
# 	case when type = 1 then neev_A when type = 2 then brev_A when type = 3 then brde_A when type = 4 then neev_A when type = 5 then cunn_A when type = 6 then euca_A when type = 7 then nede_A when type = 8 then neev_A when type = 9 then pinu_A when type = 10 then brde_A when type = 11 then neev_A when type = 12 then brde_A when type = 13 then brev_A when type = 14 then brde_A when type = 15 then neev_A end as selected_genus_A,
# 	case when type = 1 then neev_k when type = 2 then brev_k when type = 3 then brde_k when type = 4 then neev_k when type = 5 then cunn_k when type = 6 then euca_k when type = 7 then nede_k when type = 8 then neev_k when type = 9 then pinu_k when type = 10 then brde_k when type = 11 then neev_k when type = 12 then brde_k when type = 13 then brev_k when type = 14 then brde_k when type = 15 then neev_k end as selected_genus_k from filtered)
# 	select count(*) as row_count, sum(case when type is null or type != type then 1 else 0 end) as type_missing,
# 		sum(case when selected_genus_A is null or selected_genus_A != selected_genus_A then 1 else 0 end) as selected_genus_A_missing,
# 		sum(case when selected_genus_k is null or selected_genus_k != selected_genus_k then 1 else 0 end) as selected_genus_k_missing from prepared""", databaseFileName = str(OUTPUT_DIR / "Busch2024_inputs.sqlite"))

# tableToHtml("""select count(*) as row_count, sum (area) from Busch2024_dta_outputs
# 		where mi_whrv_costeff is null or mi_whrv_costeff != mi_whrv_costeff""",
# 		databaseFileName = OUTPUT_DIR / "Busch2024_dta_outputs.sqlite")

# Null area per column in Busch2024_dta_outputs.
# tableToHtml("""select column_name, null_area from (
# 	select 'source_file' as column_name, sum(case when source_file is null then area else 0 end) as null_area from Busch2024_dta_outputs union all
# 	select 'x', sum(case when x is null or x != x then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'y', sum(case when y is null or y != y then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pxl_id', sum(case when pxl_id is null or pxl_id != pxl_id then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'area', sum(case when area is null or area != area then 1 else 0 end) from Busch2024_dta_outputs union all
# 	select 'nr_accum', sum(case when nr_accum is null or nr_accum != nr_accum then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'nr_npv_estcost', sum(case when nr_npv_estcost is null or nr_npv_estcost != nr_npv_estcost then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'crop_npv_oppcost', sum(case when crop_npv_oppcost is null or crop_npv_oppcost != crop_npv_oppcost then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'nr_bgC', sum(case when nr_bgC is null or nr_bgC != nr_bgC then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'nr_soilC', sum(case when nr_soilC is null or nr_soilC != nr_soilC then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_stock', sum(case when pl_stock is null or pl_stock != pl_stock then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_bgC_whrv', sum(case when pl_bgC_whrv is null or pl_bgC_whrv != pl_bgC_whrv then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_soilC_whrv', sum(case when pl_soilC_whrv is null or pl_soilC_whrv != pl_soilC_whrv then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_npv_estcost', sum(case when pl_npv_estcost is null or pl_npv_estcost != pl_npv_estcost then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_harvested', sum(case when pl_harvested is null or pl_harvested != pl_harvested then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'nr_wohrv_costeff', sum(case when nr_wohrv_costeff is null or nr_wohrv_costeff != nr_wohrv_costeff then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_wohrv_costeff', sum(case when pl_wohrv_costeff is null or pl_wohrv_costeff != pl_wohrv_costeff then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_whrv0_costeff', sum(case when pl_whrv0_costeff is null or pl_whrv0_costeff != pl_whrv0_costeff then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_whrv_costeff', sum(case when pl_whrv_costeff is null or pl_whrv_costeff != pl_whrv_costeff then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'mi_whrv0_costeff', sum(case when mi_whrv0_costeff is null or mi_whrv0_costeff != mi_whrv0_costeff then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'mi_whrv_costeff', sum(case when mi_whrv_costeff is null or mi_whrv_costeff != mi_whrv_costeff then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'mi_whrv0_bu_costeff', sum(case when mi_whrv0_bu_costeff is null or mi_whrv0_bu_costeff != mi_whrv0_bu_costeff then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'BAUrefor', sum(case when BAUrefor is null or BAUrefor != BAUrefor then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'nr_wohrv_sGr_bu_ad', sum(case when nr_wohrv_sGr_bu_ad is null or nr_wohrv_sGr_bu_ad != nr_wohrv_sGr_bu_ad then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_wohrv_sGr_bu_ad', sum(case when pl_wohrv_sGr_bu_ad is null or pl_wohrv_sGr_bu_ad != pl_wohrv_sGr_bu_ad then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_whrv0_sGr_bu_ad', sum(case when pl_whrv0_sGr_bu_ad is null or pl_whrv0_sGr_bu_ad != pl_whrv0_sGr_bu_ad then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'pl_whrv_sGr_bu_ad', sum(case when pl_whrv_sGr_bu_ad is null or pl_whrv_sGr_bu_ad != pl_whrv_sGr_bu_ad then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'mi_whrv0_sGr_bu_ad', sum(case when mi_whrv0_sGr_bu_ad is null or mi_whrv0_sGr_bu_ad != mi_whrv0_sGr_bu_ad then area else 0 end) from Busch2024_dta_outputs union all
# 	select 'country', sum(case when country is null then area else 0 end) from Busch2024_dta_outputs)""",
# 	databaseFileName = OUTPUT_DIR / "Busch2024_dta_outputs.sqlite")

# ---------------------------
# Examine Busch2024_to_SMDAMAGE.sqlite.

# Same old 39 million with crop_va_USD_per_ha_per_year blank.
# tableToHtml("""select crop_va_USD_per_ha_per_year, sum(area_ha) as total_area_ha from Undiscounted_dta_output
# 	where crop_va_USD_per_ha_per_year is null or crop_va_USD_per_ha_per_year != crop_va_USD_per_ha_per_year""", databaseFileName = str(OUTPUT_DIR / "Busch2024_to_SMDAMAGE.sqlite"))

# Get area by rotation year.
# tableToHtml("select selected_rotation_year, count(*), sum(area_ha) from Undiscounted_dta_output group by selected_rotation_year", databaseFileName = OUTPUT_DIR / "Busch2024_to_SMDAMAGE.sqlite")

# Discretize factors A and k to try to reduce the number of bids in SMDAMAGE.
# tableToHtml("""select cast(p.selected_rotation_year as integer) as rotation_year, 10*cast(round(p.selected_A/10.0, 0) as integer) as A_bin,
# 	round(p.selected_k, 1) as k_bin, count(*) as pixel_count, sum(p.area_ha) as total_area_ha from Undiscounted_dta_output p
# 	group by rotation_year, A_bin, k_bin order by rotation_year asc""", optionalFileName= OUTPUT_DIR / "q1.html", databaseFileName = str(OUTPUT_DIR / "Busch2024_to_SMDAMAGE.sqlite"))

# Null area per column in Undiscounted_dta_output. Every column returns 390 million hectares or less. Suggests that sufficient data is available for SMDAMAGE?
# tableToHtml("""select column_name, null_area_ha from (
# 	select 'country' as column_name, sum(case when country is null then area_ha else 0 end) as null_area_ha from Undiscounted_dta_output union all
# 	select 'pixel_id', sum(case when pixel_id is null then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'plantation_genus', sum(case when plantation_genus is null then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'selected_option', sum(case when selected_option is null then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'selected_A', sum(case when selected_A is null or selected_A != selected_A then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'selected_k', sum(case when selected_k is null or selected_k != selected_k then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'selected_rotation_year', sum(case when selected_rotation_year is null or selected_rotation_year != selected_rotation_year then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'area_ha', sum(case when area_ha is null or area_ha != area_ha then 1 else 0 end) from Undiscounted_dta_output union all
# 	select 'crop_va_USD_per_ha_per_year', sum(case when crop_va_USD_per_ha_per_year is null or crop_va_USD_per_ha_per_year != crop_va_USD_per_ha_per_year then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'selected_establishment_cost_USD_per_ha', sum(case when selected_establishment_cost_USD_per_ha is null or selected_establishment_cost_USD_per_ha != selected_establishment_cost_USD_per_ha then area_ha else 0 end) from Undiscounted_dta_output union all
# 	select 'p_USD_per_tC_harvested', sum(case when p_USD_per_tC_harvested is null or p_USD_per_tC_harvested != p_USD_per_tC_harvested then area_ha else 0 end) from Undiscounted_dta_output)""",
# 	databaseFileName = OUTPUT_DIR / "Busch2024_to_SMDAMAGE.sqlite")

# Get annual tC_per_ha_per_year in one row per pixel for 100 natural-regeneration pixels with selected_rotation_year = 30.
# tableToHtml("""with sample_pixels as (
# 	select country, pixel_id from Undiscounted_dta_output
# 	where selected_option = 'natural_regeneration' and selected_rotation_year = 10
# 	order by country, pixel_id
# 	limit 100)
# select p.country, p.pixel_id,
# 	max(case when y.year = 1 then y.tC_per_ha_per_year end) as year_1,
# 	max(case when y.year = 2 then y.tC_per_ha_per_year end) as year_2,
# 	max(case when y.year = 3 then y.tC_per_ha_per_year end) as year_3,
# 	max(case when y.year = 4 then y.tC_per_ha_per_year end) as year_4,
# 	max(case when y.year = 5 then y.tC_per_ha_per_year end) as year_5,
# 	max(case when y.year = 6 then y.tC_per_ha_per_year end) as year_6,
# 	max(case when y.year = 7 then y.tC_per_ha_per_year end) as year_7,
# 	max(case when y.year = 8 then y.tC_per_ha_per_year end) as year_8,
# 	max(case when y.year = 9 then y.tC_per_ha_per_year end) as year_9,
# 	max(case when y.year = 10 then y.tC_per_ha_per_year end) as year_10,
# 	max(case when y.year = 11 then y.tC_per_ha_per_year end) as year_11,
# 	max(case when y.year = 12 then y.tC_per_ha_per_year end) as year_12,
# 	max(case when y.year = 13 then y.tC_per_ha_per_year end) as year_13,
# 	max(case when y.year = 14 then y.tC_per_ha_per_year end) as year_14,
# 	max(case when y.year = 15 then y.tC_per_ha_per_year end) as year_15,
# 	max(case when y.year = 16 then y.tC_per_ha_per_year end) as year_16,
# 	max(case when y.year = 17 then y.tC_per_ha_per_year end) as year_17,
# 	max(case when y.year = 18 then y.tC_per_ha_per_year end) as year_18,
# 	max(case when y.year = 19 then y.tC_per_ha_per_year end) as year_19,
# 	max(case when y.year = 20 then y.tC_per_ha_per_year end) as year_20,
# 	max(case when y.year = 21 then y.tC_per_ha_per_year end) as year_21,
# 	max(case when y.year = 22 then y.tC_per_ha_per_year end) as year_22,
# 	max(case when y.year = 23 then y.tC_per_ha_per_year end) as year_23,
# 	max(case when y.year = 24 then y.tC_per_ha_per_year end) as year_24,
# 	max(case when y.year = 25 then y.tC_per_ha_per_year end) as year_25,
# 	max(case when y.year = 26 then y.tC_per_ha_per_year end) as year_26,
# 	max(case when y.year = 27 then y.tC_per_ha_per_year end) as year_27,
# 	max(case when y.year = 28 then y.tC_per_ha_per_year end) as year_28,
# 	max(case when y.year = 29 then y.tC_per_ha_per_year end) as year_29,
# 	max(case when y.year = 30 then y.tC_per_ha_per_year end) as year_30
# from sample_pixels p
# left join tC_per_h_per_year y on y.country = p.country and y.pixel_id = p.pixel_id
# group by p.country, p.pixel_id
# order by p.country, p.pixel_id""", databaseFileName = OUTPUT_DIR / "Old Busch2024_to_SMDAMAGE.sqlite")

# Get 100 records with selected_rotation_year = 6 from Undiscounted_dta_output.
tableToHtml("select * from Undiscounted_dta_output where selected_rotation_year = 6 and cluster_index = 0 limit 100", \
			databaseFileName = OUTPUT_DIR / "Busch2024_to_SMDAMAGE.sqlite")

