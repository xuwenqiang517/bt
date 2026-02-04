from pathlib import Path
import pandas as pd
import os

class LocalCache:
    def __init__(self):
        self.cache_url=Path(__file__).resolve().parent.parent / "./cache"
        os.makedirs(self.cache_url, exist_ok=True)

        
    def set(self, file_path, pd):
        if pd is not None:
            pd.to_feather(self.cache_url / file_path)

    def get(self, file_path):
        try:
            return pd.read_feather(self.cache_url / file_path)
        except:
            return None
    
    def clean(self,prefix="",ignore=[]):
        for file in os.listdir(self.cache_url):
            if file.startswith(prefix) and file not in ignore:
                os.remove(self.cache_url / file)
    
if __name__ == "__main__":
    lc=LocalCache()
    df=pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
    lc.set("test.feather",df)
    df2=lc.get("test.feather")
    print(df2)
    lc.clean("test",ignore=["test.feather"])