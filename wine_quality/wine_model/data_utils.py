import pandas as pd
from typing import Union, Dict

def load_data(path_or_buffer: Union[str, 'io.BufferedReader']) -> pd.DataFrame:
    """
    Load CSV from a file path or a file-like buffer (Streamlit upload).
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(path_or_buffer)
    return df

def basic_info(df: pd.DataFrame) -> Dict:
    """
    Return small dictionary with dataset info useful for UI.
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.apply(lambda x: str(x)).to_dict(),
        'missing': df.isnull().sum().to_dict()
    }