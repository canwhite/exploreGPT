# Pyrightconfig.json 配置说明

## 什么是 pyrightconfig.json？

`pyrightconfig.json` 是 **Pyright/Pylance** 类型检查工具的配置文件。Pyright 是微软开发的 Python 静态类型检查器，被广泛用于 VS Code 的 Pylance 插件中。

## 为什么需要配置它？

### 1. 指定虚拟环境路径

Pyright 需要知道你的 Python 虚拟环境在哪里，才能正确解析依赖包的类型提示。

```json
{
  "venvPath": ".",
  "venv": ".venv"
}
```

- `venvPath`: 虚拟环境的父目录（相对于项目根目录）
- `venv`: 虚拟环境文件夹名称

### 2. 配置类型检查严格程度

```json
{
  "typeCheckingMode": "strict"  // "off" | "basic" | "standard" | "strict"
}
```

### 3. 排除不需要检查的目录

```json
{
  "exclude": [
    "**/node_modules",
    "**/__pycache__",
    ".git",
    "build",
    "dist"
  ]
}
```

### 4. 指定 Python 版本

```json
{
  "pythonVersion": "3.10"
}
```

## 常用配置项详解

| 配置项 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `venvPath` | string | 虚拟环境父目录 | `"."` |
| `venv` | string | 虚拟环境名称 | `".venv"` |
| `pythonVersion` | string | Python 版本 | `"3.10"` |
| `typeCheckingMode` | string | 类型检查模式 | `"strict"` |
| `exclude` | string[] | 排除的目录 | `["**/node_modules"]` |
| `extraPaths` | string[] | 额外的导入路径 | `["./src"]` |
| `useLibraryCodeForTypes` | boolean | 使用库的类型提示 | `true` |
| `reportMissingImports` | boolean | 报告缺失导入 | `true` |
| `reportMissingTypeStubs` | boolean | 报告缺失类型存根 | `false` |

## 与 pyproject.toml 的关系

### 方式一：独立的 pyrightconfig.json（推荐）

```json
{
  "venvPath": ".",
  "venv": ".venv",
  "pythonVersion": "3.10"
}
```

**优点**：
- 配置独立，职责清晰
- IDE 自动识别
- 适用于多个工具共存的场景

### 方式二：在 pyproject.toml 中配置

```toml
[tool.pyright]
venvPath = "."
venv = ".venv"
pythonVersion = "3.10"
```

**优点**：
- 所有配置集中在一个文件
- 减少项目根目录文件数量

**优先级**：
- `pyrightconfig.json` > `pyproject.toml` 中的 `[tool.pyright]`

## 本项目的配置说明

当前项目使用：

```json
{
  "venvPath": ".",
  "venv": ".venv"
}
```

**作用**：
- 告诉 Pyright 虚拟环境位于 `./.venv`
- 使类型检查器能正确识别 `requests`、`python-dotenv` 等依赖的类型提示
- 解决 "Import could not be resolved" 等类型错误

## 类型检查模式对比

| 模式 | 严格程度 | 适用场景 |
|------|----------|----------|
| `off` | 关闭 | 不使用类型检查 |
| `basic` | 低 | 仅检查语法错误 |
| `standard` | 中 | 默认模式，平衡严格度和实用性 |
| `strict` | 高 | 严格模式，要求所有类型注解 |

## 实际案例：解决类型错误

### 问题场景

```python
import os

url = os.getenv("API_URL")  # 类型: str | None
response = requests.post(url)  # 错误: str | None 不能赋值给 str
```

### 配置的作用

正确配置 `venv` 后，Pyright 能够：
1. 识别 `requests` 库的类型提示
2. 知道 `os.getenv()` 返回 `str | None`
3. 检测到 `requests.post()` 需要 `str` 参数
4. 报告类型不匹配错误

## 推荐配置模板

### 基础项目

```json
{
  "venvPath": ".",
  "venv": ".venv",
  "pythonVersion": "3.10",
  "typeCheckingMode": "standard"
}
```

### 严格项目

```json
{
  "venvPath": ".",
  "venv": ".venv",
  "pythonVersion": "3.10",
  "typeCheckingMode": "strict",
  "reportMissingImports": true,
  "reportMissingTypeStubs": false,
  "exclude": [
    "**/node_modules",
    "**/__pycache__"
  ]
}
```

## 常见问题

### Q1: 为什么 VS Code 提示 "Import could not be resolved"？

**原因**：Pyright 找不到虚拟环境或依赖包。

**解决**：
1. 确认虚拟环境已创建：`python -m venv .venv`
2. 确认依赖已安装：`pip install -r requirements.txt`
3. 配置 `pyrightconfig.json` 指定虚拟环境路径

### Q2: pyrightconfig.json 和 .vscode/settings.json 有什么区别？

| 文件 | 作用范围 | 用途 |
|------|----------|------|
| `pyrightconfig.json` | Pyright/Pylance | 类型检查配置 |
| `.vscode/settings.json` | VS Code | 编辑器配置（包括 Pylance） |

**建议**：类型检查配置放在 `pyrightconfig.json`，编辑器配置放在 `.vscode/settings.json`。

### Q3: 配置后还是报错怎么办？

1. 重启 VS Code 或重新加载窗口
2. 选择正确的 Python 解释器（Command Palette → "Python: Select Interpreter"）
3. 检查虚拟环境路径是否正确

## 参考资源

- [Pyright 官方文档](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)
- [Pylance 文档](https://github.com/microsoft/pylance-release)
- [Python 类型检查最佳实践](https://mypy.readthedocs.io/)