# Leap-C Windows 原生构建适配

针对本次提供的 `create_solver.py` 和 `acados_ocp_batch_solver.py` 制作。
使用 Windows、VS 2022 Build Tools、x64、Python 3.11+ 和 Leap-C 配套 acados。

## 安装

1. 将 ZIP 中的全部文件解压到 `D:\Study\SSY226-file\leap-c`。
   根目录应直接出现 `apply_leapc_windows_fix.py`、
   `smoke_test_leapc_windows.py`、`README_WINDOWS_FIX.md` 和 `patches` 文件夹。
   `patches` 内的两个文件只是安装载荷，使用下面的脚本放入正确位置。
2. 在 Anaconda Prompt 中执行：

```bat
conda activate "D:\ProgramData\Anaconda_Envs\leapc"
cd /d "D:\Study\SSY226-file\leap-c"
python apply_leapc_windows_fix.py
python smoke_test_leapc_windows.py
```

安装脚本先检查两个目标文件与本次上传版本是否一致，再替换。
替换前在各自原目录创建 `.before_leapc_windows_fix.bak` 备份。
已安装时会跳过，不会覆盖备份；版本不匹配则停止。
仅对进程设置测试用 acados、DLL 和临时目录路径；不修改全局环境变量。

## 修改内容

- 批量求解器新增可选的 `cmake_builder` 参数。
- 将构建器传给首次创建和后续扩容创建的单个求解器。
- Leap-C 在 Windows 下使用 VS 2022、x64 的 CMake 构建器。
- 为 MSVC 生成代码启用 `/openmp` 和 DLL 符号导出。
- 每个求解器使用各自的绝对构建目录，保留原有复用逻辑。
- 非 Windows 系统仍使用默认构建路径。

## 验证脚本

`smoke_test_leapc_windows.py` 使用一个小型离散双积分器，无绘图、GPU 或 RL 训练。
初始只分配一个求解器，输入两组初始状态，以验证批量扩容。
两个可微参数只影响终端代价，阶段代价固定。
目标包含最优值与最优第一步动作，用于检查两类反向传播路径。
最后用中心有限差分核对一个终端权重梯度。

首次运行会生成和编译代码，可能提示下载 `t_renderer.exe`。
成功标准是所有求解状态为 0、梯度有限且非零、有限差分检查通过，末尾显示：

```text
PASS: batched MPC, terminal-cost backpropagation and gradient check.
```

构建缓存保存在项目内的 `leapc_smoke_build`，临时目录为 `leapc_smoke_tmp`。
如需测试完全重新编译，可在停止 Python 后删除这两个测试目录再运行。

## 已完成和待完成的检查

本次已完成 Python 语法检查和五项无 DLL 的接口测试：构建器传递、首次创建、
批量扩容、复用参数和非 Windows 默认行为。
尚未在真实 Windows/MSVC/acados DLL 环境执行此端到端脚本。
因此补丁是针对已上传源码的构建适配，是否完整运行以 Windows 上的测试输出为准。
如果失败，请发完整终端输出；构建失败时需要包含前面的编译器错误。

官方参考：
- https://leap-c.github.io/leap-c/installation.html
- https://leap-c.github.io/leap-c/define_differentiable_mpc.html
- https://docs.acados.org/python_interface/index.html
