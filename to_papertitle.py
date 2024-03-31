import re

def process_paper_info(match):
    title = match.group(1)
    return f" - {{{{{title}}}}}"

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 定义匹配 paper info 的正则表达式模式
    pattern = r'\s-\s\*\*(.*?)\*\*\.(.*?)\(\[pdf\]\((.*?)\)\)\(\[link\]\((.*?)\)\)\.'

    # 使用正则表达式替换 paper info
    replaced_content = re.sub(pattern, process_paper_info, content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(replaced_content)

if __name__ == "__main__":
    input_file = "README.md"  # 输入的 Markdown 文件
    output_file = "README_.md"  # 输出的 Markdown 文件
    main(input_file, output_file)