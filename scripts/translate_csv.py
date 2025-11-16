"""
批量翻译CSV文件中的中文内容为英文
使用 deep-translator 库进行翻译，支持多个免费翻译服务
"""
import csv
import json
import ast
import os
from pathlib import Path
from tqdm import tqdm
from deep_translator import GoogleTranslator
import time
def parse_caption_dict(cap_str):
    """解析cap_seg列中的字典字符串"""
    try:
        return ast.literal_eval(cap_str)
    except:
        try:
            return eval(cap_str)
        except:
            print(f"解析失败: {cap_str[:100]}...")
            return None
def translate_text(text, translator, max_retries=3):
    """翻译单个文本，带重试机制"""
    if not text or not text.strip():
        return text
    for attempt in range(max_retries):
        try:
            translated = translator.translate(text)
            time.sleep(0.1)
            return translated
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"翻译失败，{wait_time}秒后重试: {e}")
                time.sleep(wait_time)
            else:
                print(f"翻译失败（已重试{max_retries}次）: {e}")
                return text
    return text
def translate_caption_dict(caption_dict, translator):
    """翻译整个caption字典"""
    if not caption_dict:
        return caption_dict
    translated = {}
    if 'global_caption' in caption_dict:
        translated['global_caption'] = translate_text(
            caption_dict['global_caption'], translator
        )
    if 'local_caption' in caption_dict:
        translated['local_caption'] = [
            translate_text(caption, translator)
            for caption in caption_dict['local_caption']
        ]
    return translated
def translate_csv(input_file, output_file, start_row=1, resume_file=None):
    """
    翻译CSV文件
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        start_row: 从第几行开始翻译（0为标题行，1为第一行数据）
        resume_file: 恢复文件路径（用于断点续传）
    """
    translator = GoogleTranslator(source='zh-CN', target='en')
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total_rows = len(rows)
    print(f"总共需要翻译 {total_rows} 行数据")
    translated_data = {}
    if resume_file and os.path.exists(resume_file):
        print(f"从恢复文件加载: {resume_file}")
        with open(resume_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                translated_data[row['url']] = row['cap_seg']
        print(f"已加载 {len(translated_data)} 条已翻译数据")
    if os.path.exists(output_file):
        print(f"检测到输出文件已存在: {output_file}")
        before_count = len(translated_data)
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['url'] not in translated_data:
                    translated_data[row['url']] = row['cap_seg']
        additional_count = len(translated_data) - before_count
        if additional_count > 0:
            print(f"从输出文件额外加载了 {additional_count} 条数据")
    translated_rows = []
    skipped = 0
    for idx, row in enumerate(tqdm(rows, desc="翻译进度")):
        url = row['url']
        if url in translated_data:
            translated_rows.append({
                'url': url,
                'cap_seg': translated_data[url]
            })
            skipped += 1
            continue
        cap_seg = row['cap_seg']
        caption_dict = parse_caption_dict(cap_seg)
        if caption_dict is None:
            translated_rows.append(row)
            continue
        try:
            translated_dict = translate_caption_dict(caption_dict, translator)
            translated_cap_seg = json.dumps(translated_dict, ensure_ascii=False)
            translated_rows.append({
                'url': url,
                'cap_seg': translated_cap_seg
            })
            if (idx + 1) % 10 == 0:
                if resume_file:
                    save_partial_results(resume_file, translated_rows)
                else:
                    save_partial_results(output_file, translated_rows)
        except Exception as e:
            print(f"\n处理第 {idx + 1} 行时出错: {e}")
            translated_rows.append(row)
    print(f"\n翻译完成！跳过了 {skipped} 条已翻译的数据")
    save_final_results(output_file, translated_rows)
    if resume_file and os.path.exists(resume_file):
        os.remove(resume_file)
        print(f"已删除恢复文件: {resume_file}")
def save_partial_results(output_file, rows):
    """保存部分结果（用于断点续传）"""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'cap_seg'])
        writer.writeheader()
        writer.writerows(rows)
def save_final_results(output_file, rows):
    """保存最终结果"""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'cap_seg'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果已保存到: {output_file}")
if __name__ == '__main__':
    import sys
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'dataset1.csv'
    output_file = base_dir / 'dataset1_en.csv'
    resume_file = base_dir / 'dataset1_en_resume.csv'
    print("=" * 60)
    print("CSV文件中文翻译工具")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("=" * 60)
    try:
        translate_csv(input_file, output_file, resume_file=resume_file)
        print("\n翻译任务完成！")
    except KeyboardInterrupt:
        print("\n\n翻译被用户中断")
        print(f"已保存部分结果到: {output_file}")
        print("下次运行时会自动从断点继续")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
