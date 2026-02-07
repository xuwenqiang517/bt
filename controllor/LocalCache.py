from pathlib import Path
import pandas as pd
import os
import msgpack

class LocalCache:
    def __init__(self):
        self.cache_url=Path(__file__).resolve().parent.parent / "./data"
        os.makedirs(self.cache_url, exist_ok=True)

    def get_csv(self, file_path):
        """
        从本地缓存中读取CSV文件
        :param file_path: 缓存文件基础路径（无后缀）
        :return: 读取的DataFrame或None（如果文件不存在）
        """
        try:
            return pd.read_csv(self.cache_url / f"{file_path}.csv")
        except FileNotFoundError:
            return None

    def set_csv(self, file_path, data):
        """
        缓存DataFrame到本地CSV文件
        :param file_path: 缓存文件基础路径（无后缀）
        :param data: 待存储DataFrame
        """
        if isinstance(data, pd.DataFrame):
            data.to_csv(self.cache_url / f"{file_path}.csv", index=False)
            # print(f"缓存成功: {file_path}, 数据形状: {data.shape}")
        else:
            raise ValueError("数据类型必须为pd.DataFrame")

        
    def set(self, file_path, data):
        """
        缓存数据到本地，根据数据类型自动选择存储格式
        :param file_path: 缓存文件基础路径（无后缀）
        :param data: 待存储数据（支持pd.DataFrame/dict）
        """
        if isinstance(data, pd.DataFrame) and not data.empty:
            data.to_feather(self.cache_url / f"{file_path}")
        else:
            raise ValueError("数据类型必须为pd.DataFrame")


    def get(self, file_path):
        try:
            return pd.read_feather(self.cache_url / f"{file_path}")
        except Exception as e:
            return None
        

    def clean(self,prefix="",ignore=[]):
        for file in os.listdir(self.cache_url):
            if file.startswith(prefix) and file not in ignore:
                if (self.cache_url / file).exists():
                    os.remove(self.cache_url / file)
    
if __name__ == "__main__":
    lc=LocalCache()
    df=pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
    lc.set("test__",df)
    print(f"df2:{lc.get('test__')}")
    