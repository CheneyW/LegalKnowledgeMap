

- `law_text/`初始法律文本

- `law_preprocessed/` 处理过的的数据

- `ltp_model/` ltp模型文件
- `data.json` 数据，每行为一部法律
- `format.py` 将`data.json`转化为前端易于读取的数据



`law_preprocessed/`文件夹内的文件便于观察性能，实际程序运行中并没有用到

`data.json`为实际使用的数据

预处理过的数据 和 提取概括词后的数据会相互覆盖，保存在`law_preprocessed/` 和 `data.json` 中



