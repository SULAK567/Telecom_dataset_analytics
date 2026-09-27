# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "common"))
# --- end shim ---

"""
Phase 8a — Prepare lightweight assets for the dashboard.

The district boundary file is 28 MB at full resolution. Sending that to a
browser on every interaction makes the dashboard unusable, so geometries are
simplified (topology-aware where possible) for display only. All ANALYSIS
already used the full-resolution polygons — simplification affects rendering,
never the numbers.
"""
import json

import geopandas as gpd

import config as C

OUT_STATES = C.DATA / "states_simplified.geojson"
OUT_DISTRICTS = C.DATA / "districts_simplified.geojson"


def simplify(src, key, out, tolerance):
    gdf = gpd.read_file(src)[[key, "geometry"]]
    before = src.stat().st_size / 1e6

    try:                                   # topology-preserving if available
        from topojson import Topology
        gdf = Topology(gdf, prequantize=False).toposimplify(tolerance).to_gdf()
        method = "topojson toposimplify (shared borders preserved)"
    except Exception:
        gdf["geometry"] = gdf.geometry.simplify(tolerance, preserve_topology=True)
        method = "shapely simplify(preserve_topology=True)"

    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[~gdf.geometry.is_empty]
    C.write_geojson(gdf, out)

    after = out.stat().st_size / 1e6
    print(f"{out.name}: {before:.1f} MB -> {after:.1f} MB "
          f"({after/before:.1%}) via {method}")
    print(f"  features: {len(gdf)}  tolerance: {tolerance}")
    return gdf


def main():
    print("Simplifying boundaries for dashboard rendering only.\n")
    simplify(C.STATES_GEOJSON, "st_nm", OUT_STATES, 0.02)
    simplify(C.DISTRICTS_GEOJSON, "district", OUT_DISTRICTS, 0.015)

    # sanity: every clustered region must still have a polygon to draw
    import pandas as pd
    for path, key, agg in [(OUT_STATES, "st_nm", C.STATE_CLUSTERED),
                           (OUT_DISTRICTS, "district", C.DISTRICT_CLUSTERED)]:
        g = json.loads(path.read_text())
        keys = {f["properties"][key] for f in g["features"]}
        regions = set(pd.read_csv(agg)["region"])
        missing = regions - keys
        print(f"\n{path.name}: {len(keys)} polygons; "
              f"clustered regions without a polygon: {len(missing)}")
        if missing:
            print("  " + "; ".join(sorted(missing)[:10]))


if __name__ == "__main__":
    main()
