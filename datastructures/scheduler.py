from dataclasses import dataclass
import plotly.express as px
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class Scheduler:
    variables: pd.DataFrame

    def __init__(self, file_name: str):
        self.variables = pq.read_table(file_name).to_pandas()
        assert isinstance(
            self.variables, pd.DataFrame
        ), "self.variables must be a DataFrame"
        self.variables["Start"] = self.variables["Start"].astype(float)

        self.variables = self.variables[self.variables["Start"] >= 0]  # type: ignore
        self.ready_data = self.variables[self.variables["Type"] == "Ready"]
        self.submitted_data = self.variables[(self.variables["Type"] == "Submitted")]

    @property
    def submitted_panel(self):
        return px.line(self.submitted_data, x="Start", y="Value", title="Submitted")

    @property
    def ready_panel(self):
        df_skipped = (
            pd.concat([self.ready_data.iloc[::6], self.ready_data.iloc[-1:]])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        fig = px.scatter(
            df_skipped,
            x="Start",
            y="Value",
            color="Value",
            title="Ready",
        )
        fig.update_traces(mode="lines", line_shape="hv")
        return fig
