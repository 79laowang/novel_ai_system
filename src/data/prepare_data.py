"""
中文小说数据准备脚本
支持从多种来源（TXT、JSON、JSONL）加载和预处理小说数据
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import argparse

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


@dataclass
class NovelData:
    """小说数据结构"""
    title: str
    content: str
    author: Optional[str] = None
    genre: Optional[str] = None
    tags: List[str] = None

    def to_dict(self) -> Dict:
        return {
            "title": self.title,
            "content": self.content,
            "author": self.author,
            "genre": self.genre,
            "tags": self.tags or [],
        }


class NovelDataPreparer:
    """小说数据准备器"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.raw_dir = self.data_dir / "raw"

        # 创建目录
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 标准化空白
        text = re.sub(r'\s+', ' ', text)
        # 移除过多的换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 2048,
        overlap: int = 200,
    ) -> List[str]:
        """将文本分割成训练块"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]

            # 尝试在句子边界分割
            if end < text_len:
                last_period = chunk.rfind('。')
                last_newline = chunk.rfind('\n')
                split_pos = max(last_period, last_newline)

                if split_pos > chunk_size * 0.7:  # 至少保留70%
                    chunk = text[start:start + split_pos + 1]
                    end = start + split_pos + 1

            chunks.append(chunk.strip())
            start = end - overlap if end < text_len else end

        return [c for c in chunks if len(c) > 100]

    def load_from_txt(self, txt_path: Path) -> List[NovelData]:
        """从TXT文件加载"""
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = self.clean_text(content)

        return [NovelData(
            title=txt_path.stem,
            content=content,
            genre="unknown",
        )]

    def load_from_json(self, json_path: Path) -> List[NovelData]:
        """从JSON文件加载"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        novels = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    novels.append(NovelData(
                        title=item.get('title', 'untitled'),
                        content=item.get('content', item.get('text', '')),
                        author=item.get('author'),
                        genre=item.get('genre'),
                        tags=item.get('tags', []),
                    ))
                elif isinstance(item, str):
                    novels.append(NovelData(
                        title=json_path.stem,
                        content=item,
                    ))
        elif isinstance(data, dict):
            novels.append(NovelData(
                title=data.get('title', json_path.stem),
                content=data.get('content', data.get('text', '')),
                author=data.get('author'),
                genre=data.get('genre'),
                tags=data.get('tags', []),
            ))

        return novels

    def load_from_jsonl(self, jsonl_path: Path) -> List[NovelData]:
        """从JSONL文件加载"""
        novels = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    novels.append(NovelData(
                        title=data.get('title', jsonl_path.stem),
                        content=data.get('content', data.get('text', '')),
                        author=data.get('author'),
                        genre=data.get('genre'),
                        tags=data.get('tags', []),
                    ))

        return novels

    def load_from_directory(self, dir_path: Path) -> List[NovelData]:
        """从目录加载所有文件"""
        all_novels = []

        # 支持的文件格式
        file_loaders = {
            '.txt': self.load_from_txt,
            '.json': self.load_from_json,
            '.jsonl': self.load_from_jsonl,
        }

        files = list(dir_path.rglob('*'))
        files = [f for f in files if f.is_file() and f.suffix in file_loaders]

        console.print(f"[cyan]在 {dir_path} 中找到 {len(files)} 个文件[/cyan]")

        for file_path in files:
            try:
                loader = file_loaders[file_path.suffix]
                novels = loader(file_path)
                all_novels.extend(novels)
                console.print(f"[green]✓ 加载: {file_path.name} ({len(novels)} 条记录)[/green]")
            except Exception as e:
                console.print(f"[red]✗ 加载失败: {file_path.name} - {e}[/red]")

        return all_novels

    def prepare_training_data(
        self,
        chunk_size: int = 2048,
        val_split: float = 0.1,
        min_length: int = 500,
    ) -> Dict[str, int]:
        """准备训练数据"""
        console.print("[bold cyan]开始准备训练数据...[/bold cyan]")

        # 加载原始数据
        novels = []
        if self.raw_dir.exists():
            novels.extend(self.load_from_directory(self.raw_dir))
        else:
            console.print("[yellow]未找到原始数据目录，请将小说文件放入 ./data/raw/ 目录[/yellow]")
            return {"train": 0, "val": 0}

        if not novels:
            console.print("[red]没有找到任何小说数据！[/red]")
            return {"train": 0, "val": 0}

        console.print(f"\n[green]共加载 {len(novels)} 部小说[/green]")

        # 分割成训练块
        console.print("[cyan]正在分割文本...[/cyan]")
        all_chunks = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("处理中...", total=len(novels))

            for novel in novels:
                chunks = self.split_into_chunks(
                    novel.content,
                    chunk_size=chunk_size,
                    overlap=200,
                )
                # 过滤太短的块
                chunks = [c for c in chunks if len(c) >= min_length]

                for chunk in chunks:
                    all_chunks.append({
                        "text": chunk,
                        "title": novel.title,
                        "genre": novel.genre,
                    })

                progress.update(task, advance=1)

        console.print(f"[green]共生成 {len(all_chunks)} 个训练块[/green]")

        # 划分训练集和验证集
        import random
        random.shuffle(all_chunks)

        val_size = int(len(all_chunks) * val_split)
        train_chunks = all_chunks[val_size:]
        val_chunks = all_chunks[:val_size]

        # 保存为JSONL格式
        train_file = self.train_dir / "train.jsonl"
        val_file = self.val_dir / "val.jsonl"

        console.print(f"[cyan]正在保存数据...[/cyan]")

        with open(train_file, 'w', encoding='utf-8') as f:
            for chunk in train_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        with open(val_file, 'w', encoding='utf-8') as f:
            for chunk in val_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        # 更新config中的路径
        self._update_config_paths()

        console.print(f"[bold green]✓ 数据准备完成！[/bold green]")
        console.print(f"  训练集: {len(train_chunks)} 条 → {train_file}")
        console.print(f"  验证集: {len(val_chunks)} 条 → {val_file}")

        return {"train": len(train_chunks), "val": len(val_chunks)}

    def _update_config_paths(self):
        """更新配置文件中的路径"""
        config_path = Path(__file__).parent.parent.parent / "config.py"
        if not config_path.exists():
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 更新路径
        content = re.sub(
            r'train_data_path: str = "[^"]*"',
            f'train_data_path: str = "{self.train_dir / "train.jsonl"}"',
            content,
        )
        content = re.sub(
            r'val_data_path: str = "[^"]*"',
            f'val_data_path: str = "{self.val_dir / "val.jsonl"}"',
            content,
        )

        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def create_sample_data(self):
        """创建示例数据"""
        console.print("[cyan]创建示例数据...[/cyan]")

        sample_novels = [
            {
                "title": "示例武侠小说",
                "content": """
                第一章 雪山奇遇

                天山之巅，终年积雪，寒风凛冽。少年李云独自一人盘坐在一块巨石之上，双手持剑，眼神专注地凝视着剑锋。这把剑是他师父留给他的唯一遗物，名为"寒霜"，剑身通体银白，散发着逼人的寒气。

                李云今年十六岁，自幼被一位隐世高人收养。师父去世前，只告诉他一句话："当你能够真正领悟寒霜剑法第七式的时候，就下山去寻找你的身世之谜。"

                三年来，李云日夜苦练，却始终无法突破第七式的瓶颈。就在他心灰意决之际，一位浑身是血的老人从悬崖边跌落而下...

                "年轻人，救...救我..."老人虚弱地喊道。

                李云立刻收剑起身，飞身跃向悬崖边，一把抓住了老人的衣袖。老人的重量几乎将他也拖下悬崖，但他运起内力，硬生生将老人拉了上来。

                "多谢...多谢..."老人喘息着说，"我叫张无疾，有一件重要的事情要告诉你...关于你的身世..."

                李云心头一震，难道师父说的身世之谜，今天终于要有答案了吗？
                """,
                "genre": "武侠",
            },
            {
                "title": "示例科幻小说",
                "content": """
                第一章 觉醒

                公元2157年，新上海市，第78层空中花园。

                林辰从深度睡眠中醒来，视网膜上的时间显示显示现在是凌晨三点。作为赛博格公司的高级工程师，他已经连续工作了二十个小时。大脑植入的神经芯片提醒他，还有三个项目需要在明天之前完成。

                他走到落地窗前，俯瞰着这座钢铁森林。悬浮车在空中穿梭，霓虹灯在雨幕中闪烁，全息广告牌在楼宇间舞动。这是一个高度发达的时代，也是一个充满危险的时代。

                突然，他的神经芯片接收到一条加密信息："时间到了，启动协议。"

                林辰困惑不已，他不记得自己设置过这样的协议。紧接着，一段记忆如潮水般涌入脑海——他不是真正的林辰，而是一个克隆人，被植入了一段虚假的人生记忆。

                真正的林辰，早在十年前就已经去世了...

                "这不可能..."他喃喃自语，但神经芯片传递过来的信息是如此真实，包括他"本体"生前的全部记忆和最后的嘱托。

                就在这时，房间的门被强制打开，一群武装人员冲了进来...
                """,
                "genre": "科幻",
            },
        ]

        # 创建示例文件
        for novel in sample_novels:
            file_path = self.raw_dir / f"{novel['title']}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(novel['content'].strip())
            console.print(f"[green]✓ 创建示例: {file_path.name}[/green]")

        console.print("[bold green]示例数据创建完成！[/bold green]")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description="中文小说数据准备工具")
    parser.add_argument("--data-dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--chunk-size", type=int, default=2048, help="训练块大小")
    parser.add_argument("--val-split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--min-length", type=int, default=500, help="最小文本长度")
    parser.add_argument("--sample", action="store_true", help="创建示例数据")

    args = parser.parse_args()

    preparer = NovelDataPreparer(args.data_dir)

    if args.sample:
        preparer.create_sample_data()
    else:
        stats = preparer.prepare_training_data(
            chunk_size=args.chunk_size,
            val_split=args.val_split,
            min_length=args.min_length,
        )
        console.print(f"\n[bold]统计:[/bold] 训练: {stats['train']} | 验证: {stats['val']}")


if __name__ == "__main__":
    main()
