# Task: Run Commercial Agriculture
# Who to Blame: Ryan McWay, Marta Sylla, Lijing Wang

import os
import pandas as pd
import matplotlib.pyplot as plt


def read_crop_values(path: str):
    df_crop_value = pd.read_csv(path, encoding="ISO-8859-1")

    # keep only Int$ unit
    df_crop_value = df_crop_value[df_crop_value["Unit"] == "1000 Int$"]

    # drop columns ending with F
    cols_to_drop = [col for col in df_crop_value.columns if col.endswith("F")]
    df_crop_value.drop(columns=cols_to_drop, inplace=True)

    # rename columns
    old_names = [
        "Area Code","Area Code (M49)","Area","Item Code","Item",
        "Y1961","Y1962","Y1963","Y1964","Y1965",
        "Y1966","Y1967","Y1968","Y1969","Y1970","Y1971",
        "Y1972","Y1973","Y1974","Y1975","Y1976","Y1977",
        "Y1978","Y1979","Y1980","Y1981","Y1982","Y1983",
        "Y1984","Y1985","Y1986","Y1987","Y1988","Y1989",
        "Y1990","Y1991","Y1992","Y1993","Y1994","Y1995",
        "Y1996","Y1997","Y1998","Y1999","Y2000","Y2001",
        "Y2002","Y2003","Y2004","Y2005","Y2006","Y2007",
        "Y2008","Y2009","Y2010","Y2011","Y2012","Y2013",
        "Y2014","Y2015","Y2016","Y2017","Y2018","Y2019",
        "Y2020","Y2021","Y2022"
    ]
    new_names = [
        "area_code","area_code_M49","country","crop_code","crop",
        "1961","1962","1963","1964","1965",
        "1966","1967","1968","1969","1970","1971",
        "1972","1973","1974","1975","1976","1977",
        "1978","1979","1980","1981","1982","1983",
        "1984","1985","1986","1987","1988","1989",
        "1990","1991","1992","1993","1994","1995",
        "1996","1997","1998","1999","2000","2001",
        "2002","2003","2004","2005","2006","2007",
        "2008","2009","2010","2011","2012","2013",
        "2014","2015","2016","2017","2018","2019",
        "2020","2021","2022"
    ]

    rename_dict = dict(zip(old_names, new_names))
    df_crop_value.rename(columns=rename_dict, inplace=True)

    # drop unwanted crops (e.g. aggregates)
    drop_list = ["Vegetables and Fruit Primary", "Agriculture", "Cereals, primary", "Crops", "Fibre Crops Primary", "Food", "Fruit Primary", "Livestock", "Meat indigenous, total", "Meat of asses, fresh or chilled (indigenous",
                "Meat of buffalo, fresh or chilled (indigenous)", "Meat of camels, fresh or chilled (indigenous)", "Meat of camels, fresh or chilled (indigenous)", "Meat of cattle with the bone, fresh or chilled (indigenous)",
                "Meat of chickens, fresh or chilled (indigenous)", "Meat of ducks, fresh or chilled (indigenous)", "Meat of geese, fresh or chilled (indigenous)", "Meat of goat, fresh or chilled (indigenous)",
                "Meat of mules, fresh or chilled (indigenous)", "Meat of other domestic camelids, fresh or chilled (indigenous)", "Meat of pig with the bone, fresh or chilled (indigenous)",
                "Meat of pigeons and other birds n.e.c., fresh, chilled or frozen (indigenous)", "Meat of rabbits and hares, fresh or chilled (indigenous)",
                "Meat of sheep, fresh or chilled (indigenous)", "Meat of turkeys, fresh or chilled (indigenous)", "Milk, Total", "Non Food", "Oilcrops Primary", "Roots and Tubers, Total",
                "Sugar Crops Primary", "Vegetables Primary", "Raw milk of buffalo", "Raw milk of camel", "Raw milk of cattle", "Raw milk of goats", "Raw milk of sheep"]
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

    return df_crop_value


def read_crop_coefs(path: str):
    df_crop_coefs = pd.read_csv(path, delimiter=";", encoding="utf-8")

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

    return df_crop_coefs


def merge_crop_with_coefs(df_crop_value: pd.DataFrame, df_crop_coefs: pd.DataFrame):
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

    return df_crop_value


def plot_gep_years(
    df: pd.DataFrame,
    path: str,
):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["total_gep"], marker="o", linestyle="-")
    plt.title("Time Series of Commercial Agriculture (All Countries)")
    plt.xlabel("Year")
    plt.ylabel("GEP (1000 Int$)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(path, format="png")
    plt.close()


def plot_countries_gep(df: pd.DataFrame, output_dir: str = "output"):
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


def plot_year_producers(df: pd.DataFrame, output_dir: str = "output", n=10):
    os.makedirs(output_dir, exist_ok=True)

    for year in pd.unique(df["year"]):
        df_year = df[df["year"] == year].copy()
        df_top = df_year.nlargest(n, "gep")
        plt.figure(figsize=(10, 6))
        plt.bar(df_top["country"].str.replace("", ""), df_top["gep"])
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


def group_crops(df: pd.DataFrame):
    df_gep_by_year_country = df.groupby(["area_code", "country", "year"], as_index=False).agg(gep=("gep", "sum"))
    df_gep_by_year_country = df_gep_by_year_country.sort_values(by=["area_code", "year"], ascending=[True, True])
    df_gep_by_year_country["gep"] = pd.to_numeric(df_gep_by_year_country["gep"], errors="coerce")
    return df_gep_by_year_country


def group_countries(df: pd.DataFrame):
    df_gep_by_year = df.groupby("year", as_index=False).agg(gep=("gep", "sum"))
    df_gep_by_year.set_index("year", inplace=False)
    df_gep_by_year.rename(columns={"gep": "total_gep"}, inplace=True)
    df_gep_by_year.sort_values("year", inplace=True)
    return df_gep_by_year


def run(input_dir = "input", output_dir: str = "../output"):
    # 1. Read and process data
    print("Reading csv data")
    df_crop_value = read_crop_values(os.path.join(input_dir, "Value_of_Production_E_All_Data.csv"))
    df_crop_coefs = read_crop_coefs(os.path.join(input_dir, "CWON2024_crop_coef.csv"))

    # 2. Merge data
    print("Merging dataframes")
    df_gep_by_country_year_crop = merge_crop_with_coefs(df_crop_value, df_crop_coefs)
    df_gep_by_year_country = group_crops(df_gep_by_country_year_crop)
    df_gep_by_year = group_countries(df_gep_by_year_country)

    # 3. Generate output
    os.makedirs(output_dir, exist_ok=True)
    print("Saving csvs...")
    df_gep_by_country_year_crop.to_csv(os.path.join(output_dir, "gep-year-countries-crops.csv"), index=False)
    df_gep_by_year_country.to_csv(os.path.join(output_dir, "gep-years-countries.csv"), index=False)
    df_gep_by_year.to_csv(os.path.join(output_dir, "gep-years.csv"), index=False)

    print("Plotting...")
    plot_gep_years(df_gep_by_year, os.path.join(output_dir, "gep-years.png"))
    plot_year_producers(df_gep_by_year_country, os.path.join(output_dir, "years"))
    plot_countries_gep(df_gep_by_year_country, os.path.join(output_dir, "countries"))
    print("Done.")


if __name__ == "__main__":
    run()
