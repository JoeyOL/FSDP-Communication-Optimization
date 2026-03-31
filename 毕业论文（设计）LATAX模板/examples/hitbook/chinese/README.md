# HIT 本科毕业论文 LaTeX 模板使用指南

> 基于 `hithesisbook` v3.1d 文档类，适用于哈尔滨工业大学（含深圳校区）本科/硕士/博士毕业论文排版。

---

## 一、环境配置

### 1.1 安装 TeX Live

本模板要求使用 **XeLaTeX** 编译引擎（支持中文 CJK 排版），推荐安装 **TeX Live 2026** 或更新版本。

#### macOS 安装步骤（无需 sudo）

```bash
# 1. 下载 TeX Live 安装包（使用 SJTU 镜像）
cd /tmp
curl -L -o install-tl-unx.tar.gz \
  https://mirrors.sjtug.sjtu.edu.cn/CTAN/systems/texlive/tlnet/install-tl-unx.tar.gz

# 2. 解压
tar -xzf install-tl-unx.tar.gz
cd install-tl-*

# 3. 创建安装配置（安装到用户目录，无需 root）
cat > texlive.profile << 'EOF'
selected_scheme scheme-small
TEXDIR ~/texlive/2026
TEXMFLOCAL ~/texlive/texmf-local
TEXMFSYSCONFIG ~/texlive/2026/texmf-config
TEXMFSYSVAR ~/texlive/2026/texmf-var
TEXMFHOME ~/texmf
TEXMFCONFIG ~/.texlive2026/texmf-config
TEXMFVAR ~/.texlive2026/texmf-var
binary_universal-darwin 1
instopt_adjustpath 0
instopt_adjustrepo 1
instopt_letter 0
instopt_portable 0
instopt_write18_restricted 1
tlpdbopt_autobackup 1
tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0
EOF

# 4. 执行安装
perl install-tl --profile=texlive.profile \
  --repository https://mirrors.sjtug.sjtu.edu.cn/CTAN/systems/texlive/tlnet/
```

#### 配置 PATH 环境变量

将以下内容添加到 `~/.zshrc`（macOS 默认 shell）：

```bash
echo '# TeX Live 2026' >> ~/.zshrc
echo 'export PATH="$HOME/texlive/2026/bin/universal-darwin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 验证安装

```bash
xelatex --version   # 应显示 XeTeX 0.999998 或更高版本
bibtex --version     # 应显示 BibTeX 0.99e 或更高
tlmgr --version      # TeX Live 包管理器
```

### 1.2 安装必要的 LaTeX 宏包

`scheme-small` 默认不包含本模板所需的全部宏包，需要手动补装：

```bash
tlmgr install \
  ctex xecjk cjkpunct zhnumber fandol \
  ntheorem footmisc placeins ccaption \
  splitindex perpage bigfoot \
  newtxtext newtxmath rsfs \
  gbt7714 bibunits \
  enumitem pdfpages \
  siunitx listings xcolor \
  bm mathrsfs rotating \
  ulem environ trimspaces soul \
  tocloft titletoc \
  notoccite
```

> **提示**：如果编译时提示缺少某个 `.sty` 或 `.cls` 文件，可通过以下命令查找并安装对应宏包：
> ```bash
> tlmgr search --global --file "缺少的文件名.sty"
> tlmgr install 宏包名
> ```

### 1.3 安装 Ghostscript（可选，处理 EPS 图片）

模板中部分图片使用 EPS 格式，需要 Ghostscript 进行 EPS→PDF 转换：

```bash
# macOS（通过 Homebrew）
brew install ghostscript

# 验证
gs --version
```

> 如果不安装 Ghostscript，可以手动将 EPS 图片转换为 PDF 格式放入对应目录，编译同样可以通过。

---

## 二、文件结构与编辑指南

### 2.1 目录结构总览

```
chinese/
├── thesis.tex              # ★ 主入口文件（控制文档结构）
├── front/
│   ├── cover.tex           # ★ 封面、个人信息、摘要、关键词
│   └── denotation.tex      #   符号对照表（默认未启用）
├── body/
│   ├── preface.tex          #   前言章节
│   ├── regu.tex             #   章节内容（格式规范示例）
│   ├── introduction.tex     #   章节内容（绪论示例）
│   └── name.tex             #   新增章节内容
├── back/
│   ├── conclusion.tex       # ★ 结论
│   ├── acknowledgements.tex # ★ 致谢
│   ├── publications.tex     #   攻读学位期间发表的论文
│   ├── appA.tex             #   附录（默认未启用）
│   └── ...                  #   其他后置内容
├── figures/                 #   图片目录
├── reference.bib            # ★ 参考文献数据库
├── hithesisbook.cls         #   文档类（勿修改）
├── hithesis.sty             #   附加样式包（勿修改）
├── hithesisbook.cfg         #   文档类配置（勿修改）
├── latexmkrc                #   latexmk 自动化配置
├── Makefile                 #   Make 构建配置
└── *.bst                    #   参考文献样式文件（勿修改）
```

> **★ 标记** = 你需要重点编辑的文件

### 2.2 各文件修改指南

#### （1）`thesis.tex` — 主入口文件

**作用**：控制整篇论文的结构编排，决定包含哪些章节、以什么顺序出现。

**常见操作**：

| 操作 | 方法 |
|------|------|
| 新增一个章节 | 在 `body/` 下创建 `newchapter.tex`，然后在 `\mainmatter` 区域添加 `\include{body/newchapter}` |
| 删除/隐藏章节 | 在对应的 `\include{...}` 行前加 `%` 注释 |
| 启用符号对照表 | 取消 `\input{front/denotation}` 前的 `%` 注释 |
| 启用附录 | 取消 `\begin{appendix}` 和 `\input{back/appA.tex}` 的注释 |
| 启用索引 | 取消 `\include{back/ceindex}` 的注释 |

**文档类选项**（在 `\documentclass[...]` 中修改）：

| 选项 | 可选值 | 说明 |
|------|--------|------|
| `type` | `bachelor`, `master`, `doctor`, `postdoc` | 论文类型 |
| `campus` | `harbin`, `shenzhen`, `weihai` | 校区（影响封面样式） |
| `fontset` | `fandol`, `windows`, `mac` | 字体集（`fandol` 为开源免费字体，推荐跨平台使用） |
| `chapterbold` | `true`, `false` | 章节标题是否加粗 |
| `tocblank` | `true`, `false` | 目录中章节间是否空行 |

**示例**：深圳校区硕士论文配置：
```latex
\documentclass[fontset=fandol,type=master,campus=shenzhen]{hithesisbook}
```

---

#### （2）`front/cover.tex` — 封面与摘要 ⭐最先修改

**作用**：包含个人信息（姓名、学号、导师等）和中英文摘要。

**`\hitsetup{...}` 中的关键字段**：

| 字段 | 说明 | 示例 |
|------|------|------|
| `ctitleone` | 中文标题第一行（本科封面用） | `{基于PyTorch FSDP的分布式}` |
| `ctitletwo` | 中文标题第二行（本科封面用） | `{训练通信压缩技术研究}` |
| `ctitlecover` | 封面完整中文标题 | `{基于PyTorch FSDP的分布式训练通信压缩技术研究}` |
| `ctitle` | 原创性声明中的中文标题 | 同上 |
| `cxueke` | 学科门类 | `{工学}` |
| `csubject` | 专业名称 | `{计算机科学与技术}` |
| `caffil` | 院系全称 | `{深圳校区计算机科学与技术学院}` |
| `cauthor` | 作者姓名 | `{张三}` |
| `csupervisor` | 指导教师 | `{王鸿鹏教授}` |
| `cassosupervisor` | 副指导教师（可选） | `{夏文教授}` |
| `cdate` | 答辩日期 | `{2026年6月}` |
| `firstpagecdate` | 封面首页底部日期 | `{2026年5月}` |
| `cstudentid` | 学号 | `{21B903000}` |
| `ckeywords` | 中文关键词（逗号分隔） | `{分布式训练, 通信压缩, FSDP}` |
| `ekeywords` | 英文关键词（逗号分隔） | `{distributed training, communication}` |
| `etitle` | 英文标题 | `{Research on ...}` |

**中文摘要**：修改 `\begin{cabstract}` 与 `\end{cabstract}` 之间的内容。

**英文摘要**：修改 `\begin{eabstract}` 与 `\end{eabstract}` 之间的内容。

---

#### （3）`body/*.tex` — 正文章节

**作用**：每个 `.tex` 文件对应论文的一个章节。

**编写方式**：

```latex
\chapter{绪论}

\section{研究背景}
这里写第一节的内容...

\section{国内外研究现状}

\subsection{国内研究现状}
这里写小节内容...

\subsection{国外研究现状}
...

\section{本文研究内容与结构安排}
...
```

**插入图片**：

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{your-image-name}  % 不需要写扩展名
  \caption{图片标题}
  \label{fig:your-label}
\end{figure}
```

> 图片文件放入 `figures/` 目录，推荐使用 **PDF** 或 **PNG** 格式。

**插入表格**：

```latex
\begin{table}[htbp]
  \centering
  \caption{表格标题}
  \label{tab:your-label}
  \begin{tabular}{ccc}
    \hline
    列1 & 列2 & 列3 \\
    \hline
    数据 & 数据 & 数据 \\
    \hline
  \end{tabular}
\end{table}
```

**插入公式**：

```latex
% 行内公式
梯度 $g_i$ 经过量化后...

% 独立公式（带编号）
\begin{equation}
  Q(g) = \|g\| \cdot \text{sign}(g) \cdot \xi_i
  \label{eq:qsgd}
\end{equation}

% 引用公式
如公式~\eqref{eq:qsgd} 所示...
```

**交叉引用**：

```latex
如图~\ref{fig:your-label} 所示...
如表~\ref{tab:your-label} 所示...
详见第~\ref{chap:intro} 章...
```

---

#### （4）`reference.bib` — 参考文献

**作用**：BibTeX 格式的参考文献数据库。

**添加文献条目示例**：

```bibtex
@article{alistarh2017qsgd,
  title     = {QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding},
  author    = {Alistarh, Dan and Grubic, Demjan and Li, Jerry and Tomioka, Ryota and Vojnovic, Milan},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}

@inproceedings{lin2018deep,
  title     = {Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training},
  author    = {Lin, Yujun and Han, Song and Mao, Huizi and Wang, Yu and Dally, William J},
  booktitle = {International Conference on Learning Representations},
  year      = {2018}
}
```

**在正文中引用**：

```latex
QSGD~\cite{alistarh2017qsgd} 通过随机量化实现梯度压缩...
多项研究~\cite{alistarh2017qsgd,lin2018deep} 表明...
```

> 参考文献样式为 `gbt7714-numerical`（国标 GB/T 7714 顺序编码制），无需手动调整格式。

---

#### （5）`back/conclusion.tex` — 结论

```latex
\chapter{结论}
本文针对 PyTorch FSDP 分布式训练中的通信瓶颈问题...
```

---

#### （6）`back/acknowledgements.tex` — 致谢

```latex
\chapter{致谢}
感谢我的导师...
```

---

#### （7）`back/publications.tex` — 攻读学位期间发表的论文

按学校要求的格式列出发表的论文。

---

#### （8）`figures/` — 图片目录

- 推荐格式：**PDF**（矢量图）、**PNG**（位图）
- 文件名建议使用英文，不含空格
- 在 `\includegraphics` 中引用时**不需要写扩展名**，LaTeX 会自动查找

---

## 三、编译方法

### 3.1 手动编译（4 步完整编译）

```bash
cd 毕业论文（设计）LATAX模板/examples/hitbook/chinese/

# 第一遍：生成 .aux 文件（含引用信息）
xelatex thesis

# 处理参考文献
bibtex thesis

# 第二遍：将参考文献信息写入文档
xelatex thesis

# 第三遍：解决交叉引用
xelatex thesis
```

> **为什么要编译多遍？** LaTeX 的交叉引用（图表编号、公式编号、参考文献编号）需要多遍编译才能正确解析。如果看到 `??` 或 `[?]`，说明还需要再编译一遍。

### 3.2 使用 Make（推荐）

```bash
# 编译论文
make thesis

# 编译并打开 PDF 预览
make viewthesis

# 清理临时文件
make clean

# 清理所有生成文件（含 PDF）
make cleanall
```

### 3.3 使用 latexmk（全自动）

```bash
# 使用 Makefile 调用 latexmk
make thesis METHOD=latexmk

# 或直接使用 latexmk（会自动读取 latexmkrc 配置）
latexmk
```

> `latexmk` 会自动判断需要编译的遍数，是最省心的编译方式。

---

## 四、常见问题

### Q1：编译报错 `! LaTeX Error: File 'xxx.sty' not found`

缺少 LaTeX 宏包，使用 `tlmgr` 安装：

```bash
tlmgr search --global --file "xxx.sty"   # 查找宏包名
tlmgr install 宏包名                       # 安装
```

### Q2：编译后 PDF 中出现 `??` 或 `[?]`

需要多编译几遍，或运行 `bibtex thesis` 后再编译两遍。推荐使用 `latexmk` 自动处理。

### Q3：中文显示为方框或乱码

- 确保使用 `xelatex` 而不是 `pdflatex` 编译
- 确保 `\documentclass` 中设置了 `fontset=fandol`（或你系统中有的字体集）
- 确保 `.tex` 文件编码为 **UTF-8**

### Q4：EPS 图片编译报错

需要安装 Ghostscript（见 1.3 节）。或者将 EPS 转为 PDF：

```bash
epstopdf yourimage.eps    # 需要 Ghostscript
```

### Q5：如何切换为深圳校区模板？

修改 `thesis.tex` 中 `\documentclass` 的 `campus` 选项：

```latex
\documentclass[fontset=fandol,type=bachelor,campus=shenzhen]{hithesisbook}
```

### Q6：如何添加新的章节？

1. 在 `body/` 目录下创建新文件，如 `body/chapter3.tex`
2. 在文件中写入内容（以 `\chapter{章节标题}` 开头）
3. 在 `thesis.tex` 的 `\mainmatter` 区域添加 `\include{body/chapter3}`

### Q7：参考文献没有出现？

1. 确保在 `reference.bib` 中有对应条目
2. 确保在正文中使用了 `\cite{key}` 引用
3. 重新执行完整的 4 步编译流程

---

## 五、写作工作流建议

1. **首次使用**：先修改 `front/cover.tex` 中的个人信息和摘要
2. **替换正文**：将 `body/` 下的示例文件替换为自己的章节内容
3. **添加参考文献**：在 `reference.bib` 中录入文献条目
4. **插入图表**：将图片放入 `figures/`，在正文中引用
5. **编译检查**：每完成一个章节就编译一次，及时发现问题
6. **最终编译**：提交前执行完整 4 步编译，确保无 `??` 和 warning

---

## 六、工具推荐

| 工具 | 说明 |
|------|------|
| [VS Code](https://code.visualstudio.com/) + [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) | 推荐的编辑器 + 插件，支持实时预览、语法高亮、自动编译 |
| [Overleaf](https://www.overleaf.com/) | 在线 LaTeX 编辑器，无需本地环境 |
| [JabRef](https://www.jabref.org/) | 参考文献管理工具，可导出 `.bib` 文件 |
| [Google Scholar](https://scholar.google.com/) | 搜索论文时可直接导出 BibTeX 格式 |
| [Mathpix](https://mathpix.com/) | 截图识别数学公式，转为 LaTeX 代码 |

---

> **注意**：`hithesisbook.cls`、`hithesis.sty`、`hithesisbook.cfg`、`*.bst` 等模板核心文件**不要修改**，以确保格式符合学校规范。
