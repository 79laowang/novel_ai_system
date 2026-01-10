#!/usr/bin/env python3
"""
古龙小说转换为训练数据脚本
将古龙小说txt文件转换为JSONL格式的训练数据
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

console = Console()


@dataclass
class NovelInfo:
    """小说信息"""
    title: str
    author: str
    content: str
    description: Optional[str] = None


class GulongNovelConverter:
    """古龙小说转换器"""

    def __init__(
        self,
        source_dir: str,
        output_dir: str = "./data/raw",
        author: str = "古龙",
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.author = author
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_novel_content(self, text: str) -> str:
        """提取小说正文内容"""
        lines = text.split('\n')

        # 查找正文开始标记
        start_idx = 0
        for i, line in enumerate(lines):
            if '----------小说正文开始----------' in line or '正文开始' in line:
                start_idx = i + 1
                break

        # 查找正文结束标记
        end_idx = len(lines)
        for i, line in enumerate(lines[start_idx:], start=start_idx):
            if '----------小说正文结束----------' in line or '正文结束' in line:
                end_idx = i
                break

        # 提取正文
        content_lines = lines[start_idx:end_idx]

        # 过滤掉广告、声明等无关内容
        filtered_lines = []
        skip_patterns = [
            r'武侠书库',
            r'wuxiashuku',
            r' TXT ',
            r'下载地址',
            r'本文由',
            r'请勿用于商业',
            r'整理：',
            r'书友友情提供',
        ]

        for line in content_lines:
            line = line.strip()
            if not line:
                continue
            # 跳过包含广告关键词的行
            if any(re.search(pattern, line) for pattern in skip_patterns):
                continue
            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def extract_title_and_description(self, text: str) -> tuple[str, str]:
        """提取书名和简介"""
        lines = text.split('\n')

        title = None
        description = None
        description_lines = []

        for i, line in enumerate(lines):
            line = line.strip()

            # 提取书名（去除书名号）
            if line.startswith('《') and line.endswith('》'):
                title = line.strip('《》')

            # 提取作者
            elif line.startswith('原著：') or line.startswith('作者：'):
                self.author = line.split('：', 1)[-1].strip()

            # 提取简介
            elif line.startswith('小说简介：') or line.startswith('简介：'):
                # 收集后续的简介内容
                for j in range(i + 1, len(lines)):
                    desc_line = lines[j].strip()
                    # 遇到空行或广告信息时停止
                    if not desc_line or any(keyword in desc_line for keyword in ['TXT', '武侠书库', '下载地址']):
                        break
                    description_lines.append(desc_line)
                break

        description = '\n'.join(description_lines).strip() if description_lines else None

        return title or "未知", description or ""

    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除BOM标记
        text = text.replace('\ufeff', '')

        # 统一引号格式
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # 清理多余的空白字符
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()

    def process_file(self, file_path: Path) -> Optional[NovelInfo]:
        """处理单个小说文件"""
        try:
            console.print(f"[cyan]处理: {file_path.name}[/cyan]")

            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取信息
            title, description = self.extract_title_and_description(content)

            # 如果没有提取到书名，使用文件名
            if title == "未知":
                title = file_path.stem

            # 提取正文
            novel_content = self.extract_novel_content(content)
            novel_content = self.clean_text(novel_content)

            # 检查内容长度
            if len(novel_content) < 1000:
                console.print(f"[yellow]警告: {title} 内容过短 ({len(novel_content)} 字符)[/yellow]")
                return None

            return NovelInfo(
                title=title,
                author=self.author,
                content=novel_content,
                description=description,
            )

        except Exception as e:
            console.print(f"[red]错误: 处理 {file_path.name} 失败: {e}[/red]")
            return None

    def convert_all(self) -> List[NovelInfo]:
        """转换所有小说文件"""
        txt_files = list(self.source_dir.glob('*.txt'))

        if not txt_files:
            console.print("[red]错误: 未找到任何txt文件[/red]")
            return []

        console.print(f"[bold]找到 {len(txt_files)} 个txt文件[/bold]\n")

        novels = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:

            task = progress.add_task("转换中...", total=len(txt_files))

            for file_path in txt_files:
                novel = self.process_file(file_path)
                if novel:
                    novels.append(novel)
                progress.update(task, advance=1)

        console.print(f"\n[bold green]成功转换 {len(novels)} 部小说[/bold green]")
        return novels

    def save_to_jsonl(self, novels: List[NovelInfo], output_file: str = "gulong_novels.jsonl"):
        """保存为JSONL格式"""
        output_path = self.output_dir / output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            for novel in novels:
                data = {
                    "title": novel.title,
                    "author": novel.author,
                    "content": novel.content,
                    "genre": "武侠",
                    "tags": ["古龙", "武侠", "江湖"],
                }
                if novel.description:
                    data["description"] = novel.description

                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        console.print(f"[green]已保存到: {output_path}[/green]")

        # 统计信息
        total_chars = sum(len(n.content) for n in novels)
        console.print(f"[bold]统计信息:[/bold]")
        console.print(f"  小说数量: {len(novels)}")
        console.print(f"  总字符数: {total_chars:,}")
        console.print(f"  平均字符数: {total_chars // len(novels):,}")

    def save_individual_files(self, novels: List[NovelInfo]):
        """保存为单独的txt文件"""
        individual_dir = self.output_dir / "gulong"
        individual_dir.mkdir(exist_ok=True)

        for novel in novels:
            # 安全的文件名
            safe_title = re.sub(r'[<>:"/\\|?*]', '', novel.title)
            file_path = individual_dir / f"{safe_title}.txt"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {novel.title}\n")
                f.write(f"作者: {novel.author}\n")
                if novel.description:
                    f.write(f"简介: {novel.description}\n")
                f.write("\n" + "="*50 + "\n\n")
                f.write(novel.content)

        console.print(f"[green]已保存 {len(novels)} 个单独文件到: {individual_dir}[/green]")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="古龙小说转换为训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换所有古龙小说
  python scripts/convert_gulong_novels.py

  # 指定源目录
  python scripts/convert_gulong_novels.py --source /path/to/gulong

  # 保存为单独文件
  python scripts/convert_gulong_novels.py --individual
        """,
    )

    parser.add_argument(
        '--source',
        type=str,
        default='/home/kewang/work/gulong',
        help='古龙小说目录 (默认: /home/kewang/work/gulong)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/raw',
        help='输出目录 (默认: ./data/raw)',
    )
    parser.add_argument(
        '--individual',
        action='store_true',
        help='保存为单独的txt文件',
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='gulong_novels.jsonl',
        help='输出文件名 (默认: gulong_novels.jsonl)',
    )

    args = parser.parse_args()

    console.print("[bold cyan]古龙小说转换器[/bold cyan]")
    console.print("=" * 50)
    console.print(f"源目录: {args.source}")
    console.print(f"输出目录: {args.output}")
    console.print("=" * 50 + "\n")

    # 创建转换器
    converter = GulongNovelConverter(
        source_dir=args.source,
        output_dir=args.output,
    )

    # 转换小说
    novels = converter.convert_all()

    if not novels:
        console.print("[red]没有转换任何小说[/red]")
        return

    # 保存结果
    console.print("\n[cyan]保存结果...[/cyan]")
    converter.save_to_jsonl(novels, args.output_name)

    if args.individual:
        converter.save_individual_files(novels)

    console.print("\n[bold green]转换完成！[/bold green]")
    console.print(f"\n下一步: 使用以下命令准备训练数据:")
    console.print(f"  python start.py prepare")


if __name__ == "__main__":
    main()
