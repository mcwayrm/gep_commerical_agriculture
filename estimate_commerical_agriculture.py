# Task: Run Commercial Agriculture
# Who to Blame: Ryan McWay, Marta Sylla, Lijing Wang

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def read_crop_values(path: str):
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
    
    # keep only Int$ unit
    df_crop_value = df_crop_value[df_crop_value["Unit"] == "1000 Int$"]

    # drop columns ending with F
    cols_to_drop = [col for col in df_crop_value.columns if col.endswith("F")]
    df_crop_value.drop(columns=cols_to_drop, inplace=True)

    # rename columns
    old_names = [
        "Area Code","Area Code (M49)","Area","Item Code","Item"
    ] + [f"Y{y}" for y in range(1961, 2023)]
    new_names = [
        "area_code","area_code_M49","country","crop_code","crop"
    ] + [str(y) for y in range(1961, 2023)]

    rename_dict = dict(zip(old_names, new_names))
    df_crop_value.rename(columns=rename_dict, inplace=True)

    # drop unwanted crops (e.g. aggregates)
    drop_list = ["Vegetables and Fruit Primary", "Agriculture", "Cereals, primary", "Crops", "Fibre Crops Primary", "Food", "Fruit Primary", "Livestock", "Meat indigenous, total", "Meat of asses, fresh or chilled (indigenous",
                "Meat of buffalo, fresh or chilled (indigenous)", "Meat of camels, fresh or chilled (indigenous)", "Meat of camels, fresh or chilled (indigenous)", "Meat of cattle with the bone, fresh or chilled (indigenous)",
                "Meat of chickens, fresh or chilled (indigenous)", "Meat of ducks, fresh or chilled (indigenous)", "Meat of geese, fresh or chilled (indigenous)", "Meat of goat, fresh or chilled (indigenous)",
                "Meat of mules, fresh or chilled (indigenous)", "Meat of other domestic camelids, fresh or chilled (indigenous)", "Meat of pig with the bone, fresh or chilled (indigenous)",
                "Meat of pigeons and other birds n.e.c., fresh, chilled or frozen (indigenous)", "Meat of rabbits and hares, fresh or chilled (indigenous)",
                "Meat of sheep, fresh or chilled (indigenous)", "Meat of turkeys, fresh or chilled (indigenous)", "Milk, Total", "Non Food", "Oilcrops Primary", "Roots and Tubers, Total",
                "Sugar Crops Primary", "Vegetables Primary", "Raw milk of buffalo", "Raw milk of camel", "Raw milk of cattle", "Raw milk of goats", "Raw milk of sheep","Shorn wool, greasy, including fleece-washed shorn wool", "Horse meat, fresh or chilled (indigenous)"]
    df_crop_value = df_crop_value[~df_crop_value["crop"].isin(drop_list)].copy()

    # drop unwanted countries (aggregates and currently nonexisting)
    countries_to_drop = [
        'USSR','Yugoslav SFR','World','Africa','Eastern Africa','Middle Africa',
        'Northern Africa','Southern Africa','Western Africa','Americas','Northern America',
        'Central America','Caribbean','South America','Asia','Central Asia','Eastern Asia',
        'Southern Asia','South-eastern Asia','Western Asia','Europe','Eastern Europe',
        'Northern Europe','Southern Europe','Western Europe','Oceania','Australia and New Zealand',
        'Melanesia','Micronesia','Polynesia','European Union (27)','Least Developed Countries',
        'Land Locked Developing Countries','Small Island Developing States',
        'Low Income Food Deficit Countries','Net Food Importing Developing Countries'
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
    df_crop_value['gep'] = df_crop_value['gep'] * df_crop_value['rental_rate']
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


def plot_gep_years(df: pd.DataFrame, path: str):
    """
    Line plot of global total GEP over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df["year"], df["total_gep"], marker="o", linestyle="-")
    plt.title("Time Series of Commercial Agriculture (All Countries)")
    plt.xlabel("Year")
    plt.ylabel("GEP (1000 Int$)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(path, format="png")
    plt.close()
    logging.info(f"Saved global plot to {path}.")

def plot_countries_gep(df: pd.DataFrame, output_dir: str = "output"):
    """
    One line plot per country.
    """
    os.makedirs(output_dir, exist_ok=True)

    grouped = df.groupby("country", as_index=False)
    for country, group_df in grouped:
        group_df = group_df.sort_values("year")
        plt.figure(figsize=(8, 5))
        plt.plot(group_df["year"], group_df["gep"], marker="o")
        plt.title(f"GEP Time Series of Commercial Agriculture for {country}")
        plt.xlabel("Year")
        plt.ylabel("GEP (1000 Int$)")
        plt.xticks(rotation=20)
        plt.grid(True)
        outfile = os.path.join(output_dir, f"{str(country).replace(' ', '_')}.png")
        plt.savefig(outfile, format="png")
        plt.close()
    logging.info(f"Plotted {len(grouped)} countries")


def plot_year_producers(df: pd.DataFrame, output_dir: str = "output", n=10):
    os.makedirs(output_dir, exist_ok=True)

    for year in pd.unique(df["year"]):
        df_year = df[df["year"] == year].copy()
        df_top = df_year.nlargest(n, "gep")
        plt.figure(figsize=(10, 6))
        labels = df_top["country"].str.encode('ascii', 'ignore').str.decode("utf-8") # fix non-latin characters?
        plt.bar(labels, df_top["gep"])
        plt.title(f"Top {n} Commercial Agriculture Producers in {year}")
        plt.xlabel("Country")
        plt.ylabel("GEP (1000 Int$)")
        plt.xticks(rotation=20)
        plt.grid(True)
        plt.savefig(
            os.path.join(output_dir, f"{year}_top_{n}.png"),
            format="png",
        )
        plt.close()
    logging.info(f"Plotted {len(pd.unique(df["year"]))} years charts.")


def run(input_dir = "input", output_dir: str = "../output"):
    """
    Full pipeline: read, process, merge, aggregate, save CSVs and plots.
    """
    
    # 1. Read and process data
    logging.info("Reading csv data")
    try:
        df_crop_value = read_crop_values(os.path.join(input_dir, "Value_of_Production_E_All_Data.csv"))
        df_crop_coefs = read_crop_coefs(os.path.join(input_dir, "CWON2024_crop_coef.csv"))
    except Exception:
        logging.exception("Data loading failed—aborting.")
        return
    
    # 2. Merge data
    logging.info("Merging dataframes")
    df_gep_by_country_year_crop = merge_crop_with_coefs(df_crop_value, df_crop_coefs)
    df_gep_by_year_country = group_crops(df_gep_by_country_year_crop)
    df_gep_by_year = group_countries(df_gep_by_year_country)

    # 3. Generate output
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Saving csvs...")
    df_gep_by_country_year_crop.to_csv(os.path.join(output_dir, "gep-year-countries-crops.csv"), index=False)
    logging.info(f"Saved gep-year-countries-crops.csv. ({df_gep_by_country_year_crop.shape[0]} rows).")

    df_gep_by_year_country.to_csv(os.path.join(output_dir, "gep-years-countries.csv"), index=False)
    logging.info(f"Saved gep-years-countries.csv. ({df_gep_by_year_country.shape[0]} rows).")
    
    df_gep_by_year.to_csv(os.path.join(output_dir, "gep-years.csv"), index=False)
    logging.info(f"Saved gep-years.csv. ({df_gep_by_year.shape[0]} rows).")

    logging.info("Plotting...")
    plot_gep_years(df_gep_by_year, os.path.join(output_dir, "gep-years.png"))
    plot_year_producers(df_gep_by_year_country, os.path.join(output_dir, "years"))
    plot_countries_gep(df_gep_by_year_country, os.path.join(output_dir, "countries"))
    logging.info("Run complete.")


if __name__ == "__main__":
    run()
