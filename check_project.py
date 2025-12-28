"""
项目完整性检查脚本
验证项目结构和代码语法
"""
import sys
import py_compile
import os
from pathlib import Path


def check_syntax(filepath):
    """检查Python文件语法"""
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def main():
    """主函数"""
    project_root = Path(__file__).parent
    
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "项目完整性检查" + " " * 44 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # 检查Python文件语法
    print("\n[1] Python语法检查")
    print("─" * 80)
    
    python_files = [
        'train.py',
        'evaluate.py',
        'src/data/preprocessor.py',
        'src/data/vocab.py',
        'src/data/dataset.py',
        'src/models/rnn/encoder.py',
        'src/models/rnn/decoder.py',
        'src/models/rnn/attention.py',
        'src/models/rnn/seq2seq.py',
        'src/training/trainer.py',
        'src/decoding/decoder_strategy.py',
        'src/utils/metrics.py',
    ]
    
    all_passed = True
    for file_path in python_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"  ❌ {file_path:<50} [文件不存在]")
            all_passed = False
            continue
        
        success, error = check_syntax(str(full_path))
        if success:
            print(f"  ✓ {file_path:<50} [语法正确]")
        else:
            print(f"  ❌ {file_path:<50} [语法错误]")
            all_passed = False
    
    # 检查配置文件
    print(f"\n[2] 配置文件检查")
    print("─" * 80)
    
    config_file = project_root / 'config' / 'config.yaml'
    if config_file.exists():
        print(f"  ✓ config/config.yaml{' ' * 38}[存在]")
    else:
        print(f"  ❌ config/config.yaml{' ' * 38}[不存在]")
        all_passed = False
    
    # 检查目录结构
    print(f"\n[3] 目录结构检查")
    print("─" * 80)
    
    required_dirs = [
        ('src/data', '数据处理模块'),
        ('src/models/rnn', 'RNN模型模块'),
        ('src/training', '训练模块'),
        ('src/decoding', '解码模块'),
        ('src/utils', '工具模块'),
        ('config', '配置目录'),
    ]
    
    for dir_path, desc in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            print(f"  ✓ {dir_path:<30} {desc:<20} [存在]")
        else:
            print(f"  ❌ {dir_path:<30} {desc:<20} [不存在]")
            all_passed = False
    
    # 检查文档
    print(f"\n[4] 文档检查")
    print("─" * 80)
    
    docs = [
        ('README.md', '项目说明文档'),
        ('QUICKSTART.md', '快速开始指南'),
        ('PROJECT_SUMMARY.md', '项目总结文档'),
        ('CHECKLIST.md', '功能检查清单'),
        ('requirements.txt', '依赖列表'),
    ]
    
    for doc_file, desc in docs:
        full_path = project_root / doc_file
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✓ {doc_file:<30} {desc:<20} [{size} bytes]")
        else:
            print(f"  ❌ {doc_file:<30} {desc:<20} [不存在]")
            all_passed = False
    
    # 统计信息
    print(f"\n[5] 项目统计")
    print("─" * 80)
    
    # 统计Python文件
    py_files = list(project_root.rglob("*.py"))
    py_files = [f for f in py_files if '__pycache__' not in str(f)]
    
    # 统计代码行数
    total_lines = 0
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                total_lines += len(f.readlines())
        except:
            pass
    
    print(f"  • Python文件数量: {len(py_files)}")
    print(f"  • 代码总行数: {total_lines}")
    print(f"  • 主要模块数: 5")
    print(f"  • 支持的模型: RNN (LSTM/GRU)")
    print(f"  • 注意力机制: 3种 (Dot, General, Additive)")
    print(f"  • 解码策略: 2种 (Greedy, Beam Search)")
    
    # 最终结果
    print("\n" + "╔" + "═" * 78 + "╗")
    if all_passed:
        print("║" + " " * 10 + "✅ 所有检查通过！项目结构完整，代码语法正确。" + " " * 16 + "║")
        print("╚" + "═" * 78 + "╝")
        print("\n📖 下一步：")
        print("  1. 安装依赖：pip install -r requirements.txt")
        print("  2. 查看文档：cat QUICKSTART.md")
        print("  3. 训练模型：python train.py")
        print("  4. 评估模型：python evaluate.py")
    else:
        print("║" + " " * 10 + "❌ 部分检查未通过，请修复上述问题。" + " " * 23 + "║")
        print("╚" + "═" * 78 + "╝")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

