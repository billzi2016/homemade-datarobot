"""
任务表单定义。

第一版不做极细的前端交互组件，先用 Django Form 把关键业务参数接住，
确保可以稳定生成 task 目录和 config.yaml。
"""

from __future__ import annotations

from django import forms


class TaskCreateForm(forms.Form):
    """创建任务表单。"""

    task_name = forms.CharField(
        label="任务名",
        max_length=120,
        help_text="会自动规范化为 task_xxx 风格的小写目录名。",
    )
    data_file = forms.FileField(label="数据文件")
    target_column = forms.CharField(label="目标列", max_length=120)
    task_type = forms.ChoiceField(
        label="任务类型",
        choices=[("classification", "classification"), ("regression", "regression")],
        initial="classification",
    )
    primary_metric = forms.CharField(label="主指标", initial="accuracy", max_length=64)
    cv_n_splits = forms.IntegerField(label="CV 折数", initial=3, min_value=2, max_value=10)
    test_size = forms.FloatField(label="测试集比例", initial=0.2, min_value=0.05, max_value=0.5)
    random_seed = forms.IntegerField(label="随机种子", initial=42)

    enable_analysis = forms.BooleanField(label="启用分析", required=False, initial=True)
    enable_sklearn = forms.BooleanField(label="启用 sklearn", required=False, initial=True)
    enable_torch = forms.BooleanField(label="启用 torch", required=False, initial=True)
    imbalance_enabled = forms.BooleanField(label="启用类别平衡", required=False, initial=True)

    numeric_columns = forms.CharField(
        label="数值列",
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="逗号分隔；留空表示运行时自动推断。",
    )
    categorical_columns = forms.CharField(
        label="类别列",
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="逗号分隔；留空表示运行时自动推断。",
    )
    onehot_columns = forms.CharField(
        label="One-hot 列",
        required=False,
        widget=forms.Textarea(attrs={"rows": 3}),
        help_text="逗号分隔；通常和类别列一致。",
    )

    def clean_task_name(self) -> str:
        """
        统一规范 task 名称。

        这里不让用户直接写出带空格、大小写混杂、破折号乱飞的目录名，
        否则后续文件路径、MLflow experiment name、下载链接都会变脏。
        """

        raw_name = self.cleaned_data["task_name"].strip().lower().replace("-", "_").replace(" ", "_")
        if not raw_name:
            raise forms.ValidationError("任务名不能为空。")
        if not raw_name.startswith("task_"):
            raw_name = f"task_{raw_name}"
        return raw_name
