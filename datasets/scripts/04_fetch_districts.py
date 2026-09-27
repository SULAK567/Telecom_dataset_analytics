# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
# --- end shim ---

"""
Phase 3a — Source the district boundary file.

Downloads the DataMeet Census-2011 district boundaries (641 districts) and
normalises them into data_processed/districts_india.geojson with a stable
composite key.

Source : https://github.com/datameet/maps  (DataMeet, community-maintained,
         Census 2011 district boundaries, open licensed)
Licence: CC-BY / CC0 per the DataMeet repository

Why a composite key: district NAMES are not unique in India (Aurangabad
exists in both Maharashtra and Bihar; Bilaspur in both Himachal Pradesh and
Chhattisgarh; Hamirpur in both Himachal Pradesh and Uttar Pradesh). Joining
on name alone would silently merge unrelated districts. We therefore key on
the Census code and carry a human-readable "District, State" label.

VINTAGE NOTE: these are Census 2011 boundaries, so Telangana is still inside
Andhra Pradesh here, while states_india.geojson (used at state level) has
Telangana separate. The two granularities are therefore independent views,
not a strict hierarchy. This is stated as a limitation rather than patched.
"""
import json
import urllib.request

import geopandas as gpd

import config as C

URL = "https://raw.githubusercontent.com/datameet/maps/master/docs/data/geojson/dists11.geojson"
RAW_PATH = C.DATA / "_dists11_raw.geojson"


def download():
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 1_000_000:
        print(f"Already downloaded: {RAW_PATH} ({RAW_PATH.stat().st_size/1e6:.1f} MB)")
        return
    print(f"Downloading district boundaries from:\n  {URL}")
    urllib.request.urlretrieve(URL, RAW_PATH)
    print(f"Saved: {RAW_PATH} ({RAW_PATH.stat().st_size/1e6:.1f} MB)")


def normalise():
    gdf = gpd.read_file(RAW_PATH)
    print(f"\nLoaded {len(gdf)} district polygons")
    print(f"Columns: {gdf.columns.tolist()}")

    gdf = gdf.rename(columns={"DISTRICT": "district_name", "ST_NM": "state_name"})
    gdf["district_name"] = gdf["district_name"].str.strip()
    gdf["state_name"] = gdf["state_name"].str.strip()

    # Stable unique key
    gdf["district_id"] = gdf["censuscode"].astype("Int64").astype(str)
    gdf["district"] = gdf["district_name"] + ", " + gdf["state_name"]

    dup_names = gdf["district_name"].duplicated().sum()
    dup_keys = gdf["district"].duplicated().sum()
    print(f"\nDuplicate bare district NAMES : {dup_names}  <- why we use a composite key")
    print(f"Duplicate 'District, State'   : {dup_keys}")

    if dup_keys:
        d = gdf[gdf["district"].duplicated(keep=False)].sort_values("district")
        print("Districts sharing a composite label (dissolved into one polygon):")
        print(d[["district", "district_id"]].to_string(index=False))
        gdf = gdf.dissolve(by="district", as_index=False, aggfunc="first")
        print(f"After dissolve: {len(gdf)} polygons")

    if not gdf.crs:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    # Repair any invalid polygons so the spatial join does not fail.
    invalid = (~gdf.geometry.is_valid).sum()
    print(f"\nInvalid geometries: {invalid}")
    if invalid:
        gdf["geometry"] = gdf.geometry.buffer(0)
        print(f"After buffer(0) repair: {(~gdf.geometry.is_valid).sum()} invalid")

    keep = ["district", "district_id", "district_name", "state_name", "geometry"]
    gdf = gdf[keep]

    C.write_geojson(gdf, C.DISTRICTS_GEOJSON)
    print(f"\nSAVED: {C.DISTRICTS_GEOJSON}")
    print(f"       {len(gdf)} districts across {gdf['state_name'].nunique()} states")
    print("\nSample:")
    print(gdf[["district", "district_id", "state_name"]].head(8).to_string(index=False))


if __name__ == "__main__":
    download()
    normalise()
