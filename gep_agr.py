# Task: Run Commercial Agriculture
# Who to Blame: Ryan McWay, Marta Sylla, Lijing Wang

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gep_agr_defaults

def read_crop_values(path: str, items):
    """
    Read FAO crop production values, filter by unit, drop unwanted columns/crops/countries,
    and reshape to long format.

    Returns DataFrame with columns: [area_code, country, crop_code, crop, year, gep].
    """

    try:
        df_crop_value = pd.read_csv(path, encoding="ISO-8859-1")
        logging.info(f"Loaded crop values from {path} ({df_crop_value.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop values file '{path}': {e}")
        raise

    # keep only Int$ unit AND element code 57
    df_crop_value = df_crop_value[(df_crop_value["Unit"] == "1000 USD") & (df_crop_value["Element Code"] == 57)].copy()

    # drop columns ending with F
    cols_to_drop = [col for col in df_crop_value.columns if col.endswith("F")]
    df_crop_value.drop(columns=cols_to_drop, inplace=True)

    # rename columns
    old_names = ["Area Code", "Area Code (M49)", "Area", "Item Code", "Item"] + [f"Y{y}" for y in range(1961, 2023)]
    new_names = ["area_code", "area_code_M49", "country", "crop_code", "crop"] + [str(y) for y in range(1961, 2023)]

    rename_dict = dict(zip(old_names, new_names))
    df_crop_value.rename(columns=rename_dict, inplace=True)

    # Keep only listed items
    df_crop_value = df_crop_value[df_crop_value["crop"].isin(items)].copy()

    # drop unwanted countries (aggregates and currently nonexisting)
    countries_to_drop = [
        "USSR",
        "Yugoslav SFR",
        "World",
        "Africa",
        "Eastern Africa",
        "Middle Africa",
        "Northern Africa",
        "Southern Africa",
        "Western Africa",
        "Americas",
        "Northern America",
        "Central America",
        "Caribbean",
        "South America",
        "Asia",
        "Central Asia",
        "Eastern Asia",
        "Southern Asia",
        "South-eastern Asia",
        "Western Asia",
        "Europe",
        "Eastern Europe",
        "Northern Europe",
        "Southern Europe",
        "Western Europe",
        "Oceania",
        "Australia and New Zealand",
        "Melanesia",
        "Micronesia",
        "Polynesia",
        "European Union (27)",
        "Least Developed Countries",
        "Land Locked Developing Countries",
        "Small Island Developing States",
        "Czechoslovakia" "Low Income Food Deficit Countries",
        "Net Food Importing Developing Countries",
        "China, Hong Kong SAR",
        "China, mainland",
        "China, Macao SAR",
        "China, Taiwan Province of",
        "Belgium-Luxembourg",
    ]
    df_crop_value = df_crop_value[~df_crop_value["country"].isin(countries_to_drop)]
    logging.info(f"Finished cleaning up ({df_crop_value.shape[0]} rows).")

    # reshape to long format
    df_crop_value = pd.melt(
        df_crop_value,
        id_vars=["area_code", "country", "crop_code", "crop"],
        value_vars=[str(year) for year in range(1961, 2023)],  # 1961–2022
        var_name="year",
        value_name="gep",
    )

    # ensure area_code and year are ints
    df_crop_value["area_code"] = pd.to_numeric(df_crop_value["area_code"], errors="coerce").astype(int)
    df_crop_value["year"] = pd.to_numeric(df_crop_value["year"], errors="coerce").astype(int)
    df_crop_value.loc[df_crop_value["area_code"] == 223, "country"] = "Turkey"

    logging.info(f"Reshaped to long format ({df_crop_value.shape[0]} rows).")
    return df_crop_value


def read_crop_coefs(path: str):
    """
    Read crop rental-rate coefficients, melt by decade, and build lookup table.

    Returns DataFrame with columns: [FAO, year, rental_rate].
    """
    try:
        df_crop_coefs = pd.read_csv(path, delimiter=";", encoding="utf-8")
        logging.info(f"Loaded crop coefs from {path} ({df_crop_coefs.shape[0]} rows).")
    except Exception as e:
        logging.error(f"Failed to read crop coefs file '{path}': {e}")
        raise

    df_crop_coefs = df_crop_coefs.melt(
        id_vars=["Order", "FAO", "Country/territory"],
        var_name="Decade",
        value_name="rental_rate",
    )
    df_crop_coefs["Decade_start"] = df_crop_coefs["Decade"].str.extract(r"^(\d{4})").astype(float)
    df_crop_coefs = df_crop_coefs.dropna(subset=["Decade_start"])

    # build the lookup
    df_crop_coefs = df_crop_coefs[["FAO", "Decade_start", "rental_rate"]].copy()

    # drop any rows where FAO is null (so the cast can succeed)
    df_crop_coefs = df_crop_coefs.dropna(subset=["FAO"])

    # ensure ints
    df_crop_coefs["FAO"] = df_crop_coefs["FAO"].astype(int)
    df_crop_coefs["Decade_start"] = df_crop_coefs["Decade_start"].astype(int)

    df_crop_coefs = df_crop_coefs.rename(columns={"Decade_start": "year"})
    logging.info(f"Prepared coef lookup ({df_crop_coefs.shape[0]} rows).")
    return df_crop_coefs


def merge_crop_with_coefs(df_crop_value: pd.DataFrame, df_crop_coefs: pd.DataFrame):
    """
    For each country, asof-merge crop values with rental rates by year,
    then apply rate to gep.
    """
    merged_parts = []
    for code, df_group in df_crop_value.groupby("area_code", sort=True):
        # pull the matching lookup rows for this country code
        lookup_sub = df_crop_coefs[df_crop_coefs["FAO"] == code]
        if lookup_sub.empty:
            # if no rental-rate data, fill NaN (or skip)
            df_group["rental_rate"] = pd.NA
            merged_parts.append(df_group)
            continue

        # sort within the group by year
        df_group = df_group.sort_values("year")
        lookup_sub = lookup_sub.sort_values("year")

        # now safe to asof‑merge on year only
        merged = pd.merge_asof(
            left=df_group,
            right=lookup_sub[["year", "rental_rate"]],
            on="year",
            direction="backward",
        )
        merged_parts.append(merged)

    # recombine everything
    df_crop_value = pd.concat(merged_parts, ignore_index=True)
    df_crop_value["gep"] = df_crop_value["gep"] * df_crop_value["rental_rate"]
    df_crop_value = df_crop_value.sort_values(by=["area_code", "year"], ascending=[True, True])
    logging.info(f"Merged values + coefs ({df_crop_value.shape[0]} rows).")
    return df_crop_value


def group_crops(df: pd.DataFrame):
    """
    Aggregate adjusted GEP by country-year.
    """
    df_gep_by_year_country = df.groupby(["area_code", "country", "year"], as_index=False).agg(gep=("gep", "sum"))
    df_gep_by_year_country = df_gep_by_year_country.sort_values(by=["area_code", "year"], ascending=[True, True])
    df_gep_by_year_country["gep"] = pd.to_numeric(df_gep_by_year_country["gep"], errors="coerce")
    logging.info(f"Grouped by country-year ({df_gep_by_year_country.shape[0]} rows).")
    return df_gep_by_year_country


def group_countries(df: pd.DataFrame):
    """
    Aggregate total GEP across all countries by year.
    """
    df_gep_by_year = df.groupby("year", as_index=False).agg(gep=("gep", "sum"))
    df_gep_by_year.set_index("year", inplace=False)
    df_gep_by_year.rename(columns={"gep": "total_gep"}, inplace=True)
    df_gep_by_year.sort_values("year", inplace=True)
    logging.info(f"Grouped total by year ({df_gep_by_year.shape[0]} rows).")
    return df_gep_by_year


def calculate(input_dir="input", items: list = gep_agr_defaults.DEFAULT_CROP_ITEMS):

    # 1. Read and process data
    df_crop_value = read_crop_values(os.path.join(input_dir, "Value_of_Production_E_All_Data2.csv"), items)
    df_crop_coefs = read_crop_coefs(os.path.join(input_dir, "CWON2024_crop_coef.csv"))

    df_gep_by_country_year_crop = merge_crop_with_coefs(df_crop_value, df_crop_coefs)
    df_gep_by_year_country = group_crops(df_gep_by_country_year_crop)
    df_gep_by_year = group_countries(df_gep_by_year_country)

    return {
        "gep_by_year": df_gep_by_year,
        "gep_by_year_country": df_gep_by_year_country,
        "gep_by_country_year_crop": df_gep_by_country_year_crop,
    }


if __name__ == "__main__":
    (df_gep_by_country_year_crop, df_gep_by_year_country, df_gep_by_year) = calculate("input")
