import os

def remove_single_line_comments(file_path):
    """删除单行注释和仅包含注释的行，保留代码和字符串内的#"""
    new_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # 忽略空行和仅包含注释的行
            if not stripped:
                new_lines.append(line)
                continue
            if stripped.startswith("#"):
                continue

            # 尝试删除行尾的注释，但保留字符串内的#
            new_line = ""
            i = 0
            while i < len(line):
                c = line[i]
                if c in ('"', "'"):
                    quote_char = c
                    new_line += c
                    i += 1
                    while i < len(line):
                        new_line += line[i]
                        if line[i] == quote_char and line[i-1] != '\\':
                            i += 1
                            break
                        i += 1
                elif c == '#':
                    # 遇到行内注释，停止
                    break
                else:
                    new_line += c
                    i += 1

            if new_line.strip():  # 避免删除整行后为空
                new_lines.append(new_line.rstrip() + "\n")

    # 覆盖原文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def process_directory(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                # not this file
                if full_path == __file__:
                    continue
                print(f"Processing {full_path}")
                remove_single_line_comments(full_path)

if __name__ == "__main__":
    folder_path = input("请输入要处理的文件夹路径(默认当前目录): ") or os.getcwd()
    process_directory(folder_path)
    print("处理完成。")
