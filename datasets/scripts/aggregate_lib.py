# --- shared-modules path shim (auto-added) ---
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
# --- end shim ---

"""
Shared region-aggregation logic.

The state-level and district-level pipelines are the SAME code: only the
boundary file and the grouping key change. This module holds that shared
logic so the two granularities cannot silently drift apart.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

import config as C

# Equal-area CRS used for computing region areas in km^2.
EQUAL_AREA_CRS = "EPSG:6933"


def compute_region_areas(gdf, key):
    """Return a DataFrame of [key, area_sqkm] using an equal-area projection."""
    g = gdf[[key, "geometry"]].copy()
    g = g.to_crs(EQUAL_AREA_CRS)
    g["area_sqkm"] = g.geometry.area / 1e6
    return g[[key, "area_sqkm"]].groupby(key, as_index=False)["area_sqkm"].sum()


def aggregate_towers(df, key):
    """
    Aggregate a tower-level table up to one row per region.

    Parameters
    ----------
    df  : tower-level DataFrame, must contain `key`, radio/generation,
          carrier, range, sample, created, updated
    key : name of the region column to group by (e.g. 'st_nm' or 'district')

    Returns
    -------
    DataFrame with one row per region and the full descriptive feature set.
    """
    df = df[df[key].notna()].copy()
    if hasattr(df[key], "cat"):
        df[key] = df[key].astype(str)

    g = df.groupby(key, observed=True)

    out = pd.DataFrame({"tower_count": g.size()})

    # ---- Coverage quality proxies ----
    out["range_mean"] = g["range"].mean()
    out["range_median"] = g["range"].median()
    out["sample_mean"] = g["sample"].mean()
    out["sample_total"] = g["sample"].sum()

    # ---- Recency (created/updated are UNIX timestamps) ----
    out["updated_year_mean"] = (
        pd.to_datetime(df["updated"], unit="s").dt.year.groupby(df[key], observed=True).mean()
    )
    out["created_year_mean"] = (
        pd.to_datetime(df["created"], unit="s").dt.year.groupby(df[key], observed=True).mean()
    )

    # ---- Radio generation mix (% of towers per generation) ----
    gen = (
        pd.crosstab(df[key], df["generation"], normalize="index")
        .reindex(columns=["2G", "3G", "4G", "5G"], fill_value=0.0)
        .mul(100)
    )
    gen.columns = [f"pct_{c.lower()}" for c in gen.columns]
    out = out.join(gen)

    # ---- Raw generation counts (useful for the dashboard) ----
    gen_n = (
        pd.crosstab(df[key], df["generation"])
        .reindex(columns=["2G", "3G", "4G", "5G"], fill_value=0)
    )
    gen_n.columns = [f"n_{c.lower()}" for c in gen_n.columns]
    out = out.join(gen_n)

    # ---- Carrier diversity ----
    out["carrier_count"] = g["carrier"].nunique()

    active = df[df["carrier"].isin(C.ACTIVE_CARRIERS)]
    out["active_carrier_count"] = (
        active.groupby(active[key], observed=True)["carrier"].nunique()
    )
    out["active_carrier_count"] = out["active_carrier_count"].fillna(0).astype(int)

    # ---- Carrier share (% of towers per active carrier) ----
    car = (
        pd.crosstab(df[key], df["carrier"], normalize="index")
        .reindex(columns=C.ACTIVE_CARRIERS, fill_value=0.0)
        .mul(100)
    )
    car.columns = [
        "pct_" + c.lower().replace("/", "_").replace(" ", "_") for c in car.columns
    ]
    out = out.join(car)

    # ---- Carrier counts (for dashboard charts) ----
    car_n = (
        pd.crosstab(df[key], df["carrier"])
        .reindex(columns=C.ACTIVE_CARRIERS, fill_value=0)
    )
    car_n.columns = [
        "n_" + c.lower().replace("/", "_").replace(" ", "_") for c in car_n.columns
    ]
    out = out.join(car_n)

    # ---- Market concentration: Herfindahl-Hirschman Index over carrier shares ----
    shares = pd.crosstab(df[key], df["carrier"], normalize="index")
    out["carrier_hhi"] = (shares ** 2).sum(axis=1)

    out = out.reset_index().rename(columns={key: "region"})
    return out


def attach_area_and_density(agg, areas, key):
    """Join region areas and compute density features."""
    areas = areas.rename(columns={key: "region"})
    out = agg.merge(areas, on="region", how="left")
    out["tower_density"] = out["tower_count"] / out["area_sqkm"]
    out["log_tower_density"] = np.log10(out["tower_density"].clip(lower=1e-6))
    return out


def chunked_spatial_join(csv_path, boundaries, key, dtypes, chunksize=300_000):
    """
    Spatially join a large tower CSV against boundary polygons in chunks.

    Returns a DataFrame with the original columns plus `key`. Chunking keeps
    peak memory bounded, which matters for a 2.5M-row table.
    """
    from shapely.geometry import Point

    boundaries = boundaries[[key, "geometry"]].to_crs("EPSG:4326")
    pieces = []
    total = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, dtype=dtypes, chunksize=chunksize)):
        pts = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk["long"], chunk["lat"]),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(pts, boundaries, how="left", predicate="within")
        joined = joined.drop(columns=["geometry", "index_right"])
        # A point on a shared border can match >1 polygon; keep the first.
        joined = joined[~joined.index.duplicated(keep="first")]
        pieces.append(pd.DataFrame(joined))
        total += len(chunk)
        matched = joined[key].notna().sum()
        print(f"  chunk {i+1}: {len(chunk):,} rows, {matched:,} matched "
              f"({matched/len(chunk):.2%}) | cumulative {total:,}")
    return pd.concat(pieces, ignore_index=True)
