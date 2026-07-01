# 示例数据集说明

本文档只说明当前仓库内两个示例 task 所需的数据集下载与放置方式。

## 1. 放置目录

下载完成后，请把文件放到以下位置：

- `storage/user_bizi/task_iris/data/raw/Iris.csv`
- `storage/user_bizi/task_titanic_dataset/data/raw/Titanic-Dataset.csv`

文件名请保持完全一致，否则当前示例 `config.yaml` 会直接找不到文件。

## 2. 下载来源

当前建议直接从 Kaggle 获取与示例配置匹配的 CSV。

### 2.1 Iris

目标文件名：

- `Iris.csv`

目标路径：

- `storage/user_bizi/task_iris/data/raw/Iris.csv`

列名需要与当前示例配置兼容，至少应包含：

- `Id`
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`
- `Species`

注意：

- Iris 在 Kaggle 上有很多不同版本
- 不同版本列名经常不一样
- 当前示例配置不是接受任意 iris CSV，而是要求和上面这些列名一致

### 2.2 Titanic

目标文件名：

- `Titanic-Dataset.csv`

目标路径：

- `storage/user_bizi/task_titanic_dataset/data/raw/Titanic-Dataset.csv`

列名需要与当前示例配置兼容，至少应包含：

- `PassengerId`
- `Survived`
- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Ticket`
- `Fare`
- `Cabin`
- `Embarked`

同样注意：

- Titanic 在 Kaggle 上也存在多个版本
- 当前示例配置假定使用的是包含以上列名的常见表格版本

## 3. 下载后的检查

放好文件后，可以直接核对：

- `storage/user_bizi/task_iris/data/raw/Iris.csv`
- `storage/user_bizi/task_titanic_dataset/data/raw/Titanic-Dataset.csv`

然后运行：

```bash
python3 mlflow-app/run_task.py storage/user_bizi/task_iris
python3 mlflow-app/run_task.py storage/user_bizi/task_titanic_dataset
```

## 4. 版本库策略

当前版本库只保留示例 task 的 `config.yaml`，不直接提交示例数据本体。

也就是说：

- `config.yaml` 进入 git
- `data/raw/*.csv` 不进入 git
- `outputs/`、`artifacts/`、`checkpoints/`、`run_state.json` 也不进入 git
