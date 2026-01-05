# Known Issues

本文档记录系统中已知的问题，需要后续系统分析和解决。

## 知识图谱构建问题 (Knowledge Graph Construction Issues)

### 问题描述

**日期**: 2025-01-XX  
**严重程度**: Medium  
**状态**: 待分析

在 Citation Agent 通过知识图谱检索参考文献时，发现无法正确匹配源文档的 Paper 节点。

### 问题表现

从终端输出观察到文档标识符格式不一致：

```
1. Internal Document: 2512.13564v1.pdf - internal/2512.13564v1.pdf
2. 2512.13564v1.pdf
```

### 根本原因分析（待确认）

可能的原因包括：

1. **节点 ID 格式不一致**
   - 知识图谱中存储的 Paper 节点 ID 格式：`Paper: {document_source}`
   - 从 findings 中提取的 source 格式可能为：
     - `internal/2512.13564v1.pdf`
     - `2512.13564v1.pdf`
     - `Internal Document: 2512.13564v1.pdf - internal/2512.13564v1.pdf`
   - 导致无法匹配到对应的 Paper 节点

2. **文档源标识符标准化缺失**
   - 在索引文档时（`context_manager.py`），文档源标识符可能使用完整路径或相对路径
   - 在构建知识图谱时（`graph_rag.py`），可能使用不同的标识符格式
   - 在检索时（`citation_agent.py`），又从 findings 中提取，格式可能再次不同

3. **知识图谱索引时的标识符处理**
   - `graph_rag.py` 的 `index_document` 方法中：
     ```python
     document_source = source_metadata.get("source", "unknown")
     source_doc_id = f"Paper: {document_source}"
     ```
   - 如果 `source_metadata["source"]` 的格式不一致，会导致节点 ID 不匹配

### 影响范围

- Citation Agent 无法通过知识图谱检索参考文献
- 需要回退到 RAG 检索方式
- 可能影响其他依赖知识图谱 Paper 节点的功能

### 相关代码位置

1. **知识图谱索引**:
   - `research-agent/memory/graph_rag.py:96-193` - `index_document` 方法
   - `research-agent/context_manager.py:285-373` - 文档索引流程

2. **文档源标识符提取**:
   - `research-agent/nodes/citation_agent.py:165-181` - 从 findings 提取 source

3. **Paper 节点查找**:
   - `research-agent/nodes/citation_agent.py:183-186` - 查找 Paper 节点

### 需要系统分析的问题

1. **文档源标识符的标准化流程**
   - 在文档索引时，如何统一标识符格式？
   - 在知识图谱构建时，如何确保使用相同的标识符？
   - 在检索时，如何正确匹配标识符？

2. **标识符映射机制**
   - 是否需要建立标识符映射表？
   - 是否需要支持多种格式的标识符匹配（模糊匹配）？

3. **知识图谱节点命名规范**
   - Paper 节点的 ID 格式应该是什么？
   - 是否需要支持多种格式的节点查找？

### 临时解决方案

目前 Citation Agent 已实现回退机制：
- 优先尝试知识图谱检索
- 如果失败，回退到 RAG 向量检索
- 合并两种方式的结果

### 后续工作

- [ ] 分析文档索引流程中的标识符处理
- [ ] 统一知识图谱节点命名规范
- [ ] 实现标识符标准化和映射机制
- [ ] 添加标识符格式转换工具函数
- [ ] 更新相关文档和测试

### 相关文件

- `research-agent/memory/graph_rag.py`
- `research-agent/context_manager.py`
- `research-agent/nodes/citation_agent.py`
- `research-agent/memory/graph_store.py`

