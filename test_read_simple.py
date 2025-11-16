import sys
import os
import json
try:
    import pandas as pd
    print("pandas导入成功")
except ImportError as e:
    print(f"pandas导入失败: {e}")
    print("尝试安装pandas...")
    os.system(f"{sys.executable} -m pip install pandas")
    import pandas as pd
print("\n开始读取CSV文件...")
try:
    df = pd.read_csv('dataset_en.csv')
    print(f"✓ 成功读取！形状: {df.shape}")
    print(f"✓ 列名: {df.columns.tolist()}")
    print(f"\n前2行预览:")
    for i in range(min(2, len(df))):
        print(f"\n行 {i+1}:")
        print(f"  URL: {df.iloc[i, 0][:80]}...")
        print(f"  第二列长度: {len(str(df.iloc[i, 1]))}")
        print(f"  第二列前100字符: {str(df.iloc[i, 1])[:100]}...")
    print(f"\n第二列统计:")
    print(f"  数据类型: {df.iloc[:, 1].dtype}")
    print(f"  NaN数量: {df.iloc[:, 1].isna().sum()}")
    print(f"  总行数: {len(df)}")
    print(f"\n{'='*60}")
    print("测试第二列JSON解析:")
    print(f"{'='*60}")
    success_count = 0
    fail_count = 0
    fail_rows = []
    for i in range(len(df)):
        try:
            json_str = str(df.iloc[i, 1])
            try:
                json_data = json.loads(json_str)
                success_count += 1
                if i < 3:
                    print(f"\n✓ 行 {i+1} - JSON解析成功:")
                    print(f"  global_caption: {json_data.get('global_caption', 'N/A')[:80]}...")
                    print(f"  local_caption数量: {len(json_data.get('local_caption', []))}")
            except json.JSONDecodeError as je:
                try:
                    fixed_str = json_str.replace('""', '"')
                    json_data = json.loads(fixed_str)
                    success_count += 1
                    if i < 3:
                        print(f"\n✓ 行 {i+1} - JSON解析成功（修复转义后）:")
                        print(f"  global_caption: {json_data.get('global_caption', 'N/A')[:80]}...")
                        print(f"  local_caption数量: {len(json_data.get('local_caption', []))}")
                except json.JSONDecodeError as je2:
                    fail_count += 1
                    fail_rows.append(i + 1)
                    if i < 3:
                        print(f"\n✗ 行 {i+1} - JSON解析失败:")
                        print(f"  错误: {je2}")
                        print(f"  原始字符串前200字符: {json_str[:200]}...")
        except Exception as e:
            fail_count += 1
            fail_rows.append(i + 1)
            if i < 3:
                print(f"\n✗ 行 {i+1} - 处理异常: {e}")
    print(f"\n{'='*60}")
    print(f"JSON解析统计:")
    print(f"  成功: {success_count} 行")
    print(f"  失败: {fail_count} 行")
    print(f"  成功率: {success_count/(success_count+fail_count)*100:.2f}%")
    if fail_rows:
        print(f"  失败行号（前10个）: {fail_rows[:10]}")
    print(f"{'='*60}")
except Exception as e:
    print(f"✗ 读取失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n尝试使用不同的参数读取...")
    try:
        df = pd.read_csv('dataset_en.csv', quotechar='"', escapechar='\\', doublequote=True)
        print("使用escapechar参数成功！")
    except Exception as e2:
        print(f"仍然失败: {e2}")
        try:
            df = pd.read_csv('dataset_en.csv', quoting=1)
            print("使用quoting参数成功！")
        except Exception as e3:
            print(f"仍然失败: {e3}")
