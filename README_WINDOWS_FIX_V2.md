# Leap-C Windows 原生补丁 v2

适用于本次上传的 acados C 源文件、生成模板，以及之前的两个 Python 源文件。
保留 OpenMP，使用已有 Anaconda leapc 环境和 VS 2022 x64 工具链。

## 执行顺序（Anaconda Prompt）

将 ZIP 内容直接解压到 `D:\Study\SSY226-file\leap-c`，使安装脚本直接位于根目录。
然后逐条执行：

```bat
conda activate "D:\ProgramData\Anaconda_Envs\leapc"
cd /d "D:\Study\SSY226-file\leap-c"
python apply_leapc_windows_fix_v2.py
rebuild_acados_windows_v2.cmd
cd /d "D:\Study\SSY226-file\leap-c"
python smoke_test_leapc_windows.py
```

上一条成功后再继续下一条。已有 v1 补丁也可以使用本包；不必先还原。
安装脚本先检查全部 9 个目标文件的版本和载荷，再开始替换。
每个修改文件旁边会保存 `.before_leapc_windows_fix_v2.bak` 备份；不会覆盖现有备份。
支持重复执行；遇到不同源码版本则停止，并指出对应路径。

## 具体修改

1. 5 个底层 C 文件中的 29 处 OpenMP 循环：将迭代变量声明移到 pragma 前，
   并用局部大括号保留原循环变量作用域。OpenMP 的 for/parallel for/nowait 指令保留。
2. `acados_solver.in.c` 中 6 个批量函数作同样处理，避免随后生成求解器时再次 C3015。
3. `CMakeLists.in.txt` 的 MSVC 分支使用本机 C11、OpenMP imported target 和 DLL 符号导出，
   不再让 GNU 编译标志覆盖 MSVC 设置；给已经创建的对象目标显式加入 OpenMP。
   后续共享库也链接 OpenMP 运行时。GNU 分支继续使用原编译标志。
4. 批量求解器的可选 CMake builder 传递路径，包括首次创建和扩容创建。
5. Windows 下只有生成的求解器 DLL 存在时才尝试复用缓存。
   JSON 存在但上次编译失败时会重新生成并编译。

## 编译与安装

重建脚本使用新的 `external\acados\build_msvc_v2` 构建目录，
开启 `ACADOS_WITH_OPENMP=ON`，并设置 `ACADOS_NUM_THREADS=1`。
成功后安装至该项目配套 acados 的 bin/lib/include。
脚本会检查 `bin\acados.dll` 和 `include\acados_c\ocp_nlp_interface.h`。
编译失败时日志位于 `external\acados\acados_build_v2_log.txt`。
安装失败时日志位于 `external\acados\acados_install_v2_log.txt`。

## 运行验证

配套 smoke 脚本检查批量扩容、两组初始状态求解、终端权重反向传播和中心有限差分。
成功标志为：

```text
PASS: batched MPC, terminal-cost backpropagation and gradient check.
```

测试会在当前进程设置项目配套 acados 的 DLL 路径及不含空格的临时目录，
不修改全局环境变量，不需要 GPU 或绘图。

## 验证范围

本次执行源码结构检查、补丁版本/备份/重复执行检查、Python 构建器与缓存路径接口测试，
以及使用源文件中全部 35 处循环头构造的 GCC/OpenMP 多线程等价性测试。
真实 Windows/MSVC 的 acados 完整编译和 Leap-C 端到端运行仍需在你的电脑上验证。
这些本地检查不能替代 Windows 编译结果。

编译失败请发 `acados_build_v2_log.txt`；运行失败请发完整终端输出。
