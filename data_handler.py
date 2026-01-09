import pandas as pd
import datetime
import os
import requests
import wbdata


class Data_Handler:
    @staticmethod
    def reshape_long_HDI(df, indicators: dict):
        """
        Converts wide-format year columns into long format for easier filtering.
        """
        id_vars = ['iso3', 'country', 'region']

        # Determine columns that match indicator prefixes
        value_vars = [
            col for col in df.columns
            if any(col.startswith(prefix + "_") for prefix in indicators.keys())
        ]

        long_df = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='metric_year',
            value_name='value'
        )

        # Split 'metric_year' into 'metric' and 'year'
        long_df[['metric', 'year']] = long_df['metric_year'].str.rsplit('_', n=1, expand=True)
        long_df['year'] = long_df['year'].astype(int)
        long_df.drop(columns='metric_year', inplace=True)

        # Add readable metric names
        long_df['metric_name'] = long_df['metric'].map(indicators)

        return long_df

    @staticmethod
    def get_data_HDI(filepath: str, indicators: dict, countries=None, start_year=None, end_year=None):
        """
        Retrieve filtered data based on an indicator dictionary, countries, and year range.

        Args:
            indicators (dict): Dictionary of indicator IDs and readable names.
            countries (list or str): Country or list of countries to filter by.
            start_year (int): Start year for filtering.
            end_year (int): End year for filtering.
            :param filepath: path of the HDI dataset
        """
        # Read CSV safely
        df = pd.read_csv(filepath, encoding="ISO-8859-1")

        # Standardize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        # Convert to long format using the indicators provided
        long_df = Data_Handler.reshape_long_HDI(df, indicators)

        # Filter by countries
        if countries is not None:
            if isinstance(countries, str):
                countries = [countries]
            long_df = long_df[long_df['country'].str.lower().isin([c.lower() for c in countries])]

        # Filter by year range
        if start_year is not None:
            long_df = long_df[long_df['year'] >= start_year]
        if end_year is not None:
            long_df = long_df[long_df['year'] <= end_year]

        return long_df.reset_index(drop=True)




    @staticmethod
    def get_data_IDMC(filepath: str, indicators: dict, iso3=None, start_year: int = None, end_year: int = None, hazard_category_name: str = None, hazard_type_name: str = None):
        """
        Retrieve filtered IDMC data using an indicator dictionary.
        """

        # Load data
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    
        # ---- Filtering ----
        if iso3:
            if isinstance(iso3, str):
                iso3 = [iso3]
            df = df[df["iso3"].isin(iso3)]

        if start_year:
            df = df[df["year"] >= start_year]

        if end_year:
            df = df[df["year"] <= end_year]

        if hazard_category_name:
            df = df[
                df["hazard_category_name"].str.lower()
                == hazard_category_name.lower()
            ]

        if hazard_type_name:
            df = df[
                df["hazard_type_name"].str.lower()
                == hazard_type_name.lower()
            ]

        # ---- Indicator handling ----
        base_cols = ["iso3",  "start_date", "year"]
    
        if indicators in (None, "all"):
            # keep everything except base columns duplication
            cols = base_cols + [
                c for c in df.columns if c not in base_cols
            ]
            df = df[cols]
    
        else:
            indicator_cols = list(indicators.keys())
            df = df[base_cols + indicator_cols]
            df = df.rename(columns=indicators)
    
        return df.reset_index(drop=True)

    @staticmethod
    def get_data_WB(indicators, countries="all", start_year=None, end_year=None):
        """
        Fetch World Bank data using wbdata.get_data for one or more indicators, including ISO3 codes.
        Works with the new flattened structure and includes optional date filtering.

        Parameters:
            indicators (dict): Mapping from indicator code to descriptive name,
                               e.g. {'NY.GDP.MKTP.CD': 'GDP', 'SP.POP.TOTL': 'Population'}
            countries (list or str): ISO2 country codes like ['US', 'CN'], or 'all'
            start_year (int): Start year (optional)
            end_year (int): End year (optional)

        Returns:
            pd.DataFrame: DataFrame with columns ['Country', 'ISO3', 'Year', ...indicators...]
        """
        import pandas as pd
        from functools import reduce
        import wbdata

        all_data = []

        for code, name in indicators.items():
            raw_data = wbdata.get_data(indicator=code, country=countries)

            rows = []
            for record in raw_data:
                year = int(record['date'])
                # Optional: skip missing values
                value = record.get('value')
                country_name = record['country']['value']
                country_iso3 = record.get('countryiso3code', record['country']['id'])

                rows.append({
                    "Country": country_name,
                    "ISO3": country_iso3,
                    "Year": year,
                    name: value
                })

            all_data.append(pd.DataFrame(rows))

        # Merge all indicators
        if all_data:
            df = reduce(lambda left, right: pd.merge(left, right, on=['Country', 'ISO3', 'Year'], how='outer'), all_data)
        else:
            df = pd.DataFrame(columns=['Country', 'ISO3', 'Year'] + list(indicators.values()))

        # Filter by start_year / end_year
        if start_year is not None:
            df = df[df['Year'] >= start_year]
        if end_year is not None:
            df = df[df['Year'] <= end_year]

        # Reorder columns
        cols = ['Country', 'ISO3', 'Year'] + list(indicators.values())
        df = df[[c for c in cols if c in df.columns]]

        return df



    @staticmethod
    def get_data_GIDD(client_id: str, limit: int = 500, iso3=None, start_year: str = None, end_year: str = None, hazard_category_name: str = None, hazard_type_name: str = None, indicators=None) -> pd.DataFrame:
        """
        Retrieve GIDD disaster data from the IDMC external API.
        """

        # --- Fetch all records ---
        base_url = "https://helix-tools-api.idmcdb.org/external-api/gidd/disasters/"
        params = {"client_id": client_id, "limit": limit}
        all_records = []

        while True:
            r = requests.get(base_url, params=params)
            r.raise_for_status()
            js = r.json()
            all_records.extend(js.get("results", []))

            if js.get("next"):
                base_url = js["next"]
                params = {}  # next URL has full query
            else:
                break

        # --- Convert to DataFrame ---
        df = pd.DataFrame(all_records)

        # --- Standardize column names ---
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")


        # --- Filtering ---
        if iso3 not in (None, "all"):
            if isinstance(iso3, str):
                iso3 = [iso3]
            df = df[df["iso3"].isin([c.strip() for c in iso3])]


        if start_year:
            df = df[df["year"] >= start_year]
    
        if end_year:
            df = df[df["year"] <= end_year]

        if hazard_category_name:
            df = df[df["hazard_category_name"].str.lower() == hazard_category_name.lower()]

        if hazard_type_name:
            df = df[df["hazard_type_name"].str.lower() == hazard_type_name.lower()]

        # --- Indicator selection ---
        base_cols = ["iso3", "year", "start_date"]
        if indicators in (None, "all"):
            df = df  # keep all columns
        else:
            df = df[base_cols + list(indicators)]

        return df.reset_index(drop=True)
    
            
    
