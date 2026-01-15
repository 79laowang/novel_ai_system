"""
记忆管理模块 - 使用向量数据库实现长期记忆和RAG
支持故事上下文、人物关系、情节线索的持久化存储和检索
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chromadb import Client, PersistentClient
from chromadb.config import Settings
from rich.console import Console
from rich import print as rprint

# 不在模块级别导入 SentenceTransformer，而是在需要时动态导入
# 这样可以在导入前设置 HF_ENDPOINT

console = Console()


@dataclass
class MemoryItem:
    """记忆项"""
    content: str
    memory_type: str  # context, character, plot, setting
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class NovelMemoryManager:
    """小说记忆管理器"""

    # 记忆类型
    MEMORY_TYPES = {
        "context": "故事上下文",
        "character": "人物信息",
        "plot": "情节线索",
        "setting": "环境设定",
        "dialogue": "重要对话",
    }

    def __init__(self, config):
        self.config = config
        self.client: Optional[Client] = None
        self.collections: Dict[str, Any] = {}
        self.embedding_model: Optional[SentenceTransformer] = None
        self.long_term_memory: Dict[str, Any] = {}

    def initialize(self):
        """初始化记忆系统"""
        rprint("[bold cyan]正在初始化记忆系统...[/bold cyan]")

        # 初始化ChromaDB
        persist_dir = Path(self.config.memory.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = PersistentClient(
            path=str(persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # 初始化embedding模型
        if not self.config.memory.embedding_model:
            rprint("[yellow]⚠ Embedding模型未配置，记忆功能将不可用[/yellow]")
            self.embedding_model = None
        else:
            local_model_dir = None

            # 第一次尝试：从本地 ./models/embeddings/ 加载
            local_embeddings_dir = Path("./models/embeddings/")
            if local_embeddings_dir.exists() and (local_embeddings_dir / "config.json").exists() and (local_embeddings_dir / "pytorch_model.bin").exists():
                try:
                    rprint("[cyan]加载Embedding模型（从本地 ./models/embeddings/）...[/cyan]")
                    from sentence_transformers import SentenceTransformer as ST
                    self.embedding_model = ST(str(local_embeddings_dir), device="cuda" if self._has_cuda() else "cpu")
                    rprint(f"[green]✓ Embedding模型加载成功（本地）[/green]")
                except Exception as e:
                    rprint(f"[yellow]本地模型加载失败: {e}，尝试下载...[/yellow]")
                    local_model_dir = None
            else:
                rprint("[yellow]本地模型不存在，尝试从远程下载...[/yellow]")
                local_model_dir = Path.home() / ".cache" / "huggingface" / "models" / self.config.memory.embedding_model.replace("/", "--")

                # 第二次尝试：中国镜像站
                if not local_model_dir.exists():
                    try:
                        rprint("[cyan]下载Embedding模型 (中国镜像站: hf-mirror.com)...[/cyan]")

                        # 下载模型到本地目录（显式指定endpoint）
                        from huggingface_hub import snapshot_download
                        snapshot_download(
                            repo_id=self.config.memory.embedding_model,
                            endpoint="https://hf-mirror.com",
                            local_dir=str(local_model_dir),
                            local_dir_use_symlinks=False,
                            ignore_patterns=["*.DS_Store", "*.md"],
                        )
                        rprint(f"[green]✓ 模型下载完成[/green]")
                    except Exception as e:
                        rprint(f"[yellow]中国镜像站下载失败: {e}，尝试官方站点...[/yellow]")
                        local_model_dir = Path.home() / ".cache" / "huggingface" / "models_official" / self.config.memory.embedding_model.replace("/", "--")

                        # 第三次尝试：官方站点（不指定endpoint，使用默认）
                        try:
                            snapshot_download(
                                repo_id=self.config.memory.embedding_model,
                                local_dir=str(local_model_dir),
                                local_dir_use_symlinks=False,
                                ignore_patterns=["*.DS_Store", "*.md"],
                            )
                            rprint(f"[green]✓ 模型下载完成（官方站点）[/green]")
                        except Exception as e2:
                            rprint(f"[yellow]官方站点也下载失败: {e2}[/yellow]")
                            local_model_dir = None

                # 加载模型（从下载的本地路径）
                if local_model_dir and local_model_dir.exists():
                    try:
                        rprint("[cyan]加载Embedding模型（从本地）...[/cyan]")
                        from sentence_transformers import SentenceTransformer as ST
                        self.embedding_model = ST(str(local_model_dir), device="cuda" if self._has_cuda() else "cpu")
                        rprint(f"[green]✓ Embedding模型加载成功[/green]")
                    except Exception as e:
                        rprint(f"[yellow]本地模型加载失败: {e}[/yellow]")
                        self.embedding_model = None
                else:
                    rprint("[yellow]⚠ 模型文件不存在，记忆功能将不可用[/yellow]")
                    self.embedding_model = None

        # 初始化各个记忆类型的集合
        for memory_type in self.MEMORY_TYPES:
            collection_name = f"novel_{memory_type}"
            try:
                self.collections[memory_type] = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"type": memory_type},
                )
            except Exception as e:
                rprint(f"[yellow]警告: 创建集合 {collection_name} 失败: {e}[/yellow]")

        # 加载长期记忆
        self._load_long_term_memory()

        if self.embedding_model:
            rprint(f"[green]✓ 记忆系统初始化完成，包含 {len(self.MEMORY_TYPES)} 种记忆类型[/green]")
        else:
            rprint(f"[yellow]⚠ 记忆系统部分初始化完成（embedding模型不可用）[/yellow]")

    def _has_cuda(self) -> bool:
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_long_term_memory(self):
        """加载长期记忆"""
        memory_path = Path(self.config.memory.long_term_memory_path)
        if memory_path.exists():
            with open(memory_path, 'r', encoding='utf-8') as f:
                self.long_term_memory = json.load(f)
            rprint(f"[green]✓ 加载长期记忆: {len(self.long_term_memory)} 条[/green]")

    def _save_long_term_memory(self):
        """保存长期记忆"""
        memory_path = Path(self.config.memory.long_term_memory_path)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)

    def _generate_id(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def add_memory(
        self,
        content: str,
        memory_type: str = "context",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加记忆"""
        if memory_type not in self.MEMORY_TYPES:
            raise ValueError(f"无效的记忆类型: {memory_type}")

        # 创建记忆项
        memory = MemoryItem(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
        )

        # 添加到向量数据库
        collection = self.collections[memory_type]
        memory_id = self._generate_id(content)

        collection.add(
            documents=[content],
            ids=[memory_id],
            metadatas=[memory.to_dict()],
        )

        # 添加到长期记忆
        if memory_type not in self.long_term_memory:
            self.long_term_memory[memory_type] = []
        self.long_term_memory[memory_type].append(memory.to_dict())

        # 限制记忆数量
        if len(self.long_term_memory[memory_type]) > self.config.memory.max_memory_items:
            self.long_term_memory[memory_type] = \
                self.long_term_memory[memory_type][-self.config.memory.max_memory_items:]

        self._save_long_term_memory()

        return memory_id

    def retrieve_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        top_k = top_k or self.config.memory.memory_retrieval_top_k

        results = []

        # 如果指定类型，只从该类型检索
        types_to_search = [memory_type] if memory_type else list(self.MEMORY_TYPES.keys())

        for m_type in types_to_search:
            collection = self.collections.get(m_type)
            if not collection:
                continue

            try:
                query_results = collection.query(
                    query_texts=[query],
                    n_results=top_k,
                )

                for doc, metadata, distance in zip(
                    query_results['documents'][0],
                    query_results['metadatas'][0],
                    query_results['distances'][0],
                ):
                    results.append({
                        'content': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'type': m_type,
                    })
            except Exception as e:
                console.print(f"[yellow]检索 {m_type} 失败: {e}[/yellow]")

        # 按距离排序
        results.sort(key=lambda x: x['distance'])
        return results[:top_k]

    def get_context_memory(self, session_id: str) -> str:
        """获取会话的上下文记忆"""
        # 检索最近的相关记忆
        memories = self.retrieve_memory(
            query=session_id,
            memory_type="context",
            top_k=3,
        )

        if not memories:
            return ""

        # 格式化为上下文
        context_parts = []
        for mem in memories:
            context_parts.append(f"- {mem['content']}")

        return "【故事记忆】\n" + "\n".join(context_parts)

    def get_character_memory(self, character_name: Optional[str] = None) -> str:
        """获取人物记忆"""
        query = character_name or "主要人物"
        memories = self.retrieve_memory(
            query=query,
            memory_type="character",
            top_k=5,
        )

        if not memories:
            return ""

        character_info = []
        for mem in memories:
            character_info.append(f"- {mem['content']}")

        return "【人物信息】\n" + "\n".join(character_info)

    def get_plot_memory(self) -> str:
        """获取情节记忆"""
        memories = self.retrieve_memory(
            query="情节发展",
            memory_type="plot",
            top_k=5,
        )

        if not memories:
            return ""

        plot_lines = []
        for mem in memories:
            plot_lines.append(f"- {mem['content']}")

        return "【情节线索】\n" + "\n".join(plot_lines)

    def add_character(
        self,
        name: str,
        description: str,
        personality: Optional[str] = None,
        background: Optional[str] = None,
    ) -> str:
        """添加人物记忆"""
        content = f"{name}: {description}"
        if personality:
            content += f"\n性格: {personality}"
        if background:
            content += f"\n背景: {background}"

        metadata = {
            "name": name,
            "personality": personality,
            "background": background,
        }

        return self.add_memory(content, memory_type="character", metadata=metadata)

    def add_plot(self, plot_description: str, importance: str = "normal"):
        """添加情节线索"""
        metadata = {"importance": importance}
        return self.add_memory(plot_description, memory_type="plot", metadata=metadata)

    def add_setting(self, setting_description: str):
        """添加环境设定"""
        return self.add_memory(setting_description, memory_type="setting")

    def add_dialogue(self, dialogue: str, speaker: Optional[str] = None):
        """添加重要对话"""
        metadata = {"speaker": speaker}
        return self.add_memory(dialogue, memory_type="dialogue", metadata=metadata)

    def summarize_session(self, session_content: str) -> str:
        """总结会话内容并添加到记忆"""
        # 使用简单的分割来提取关键信息
        # 实际应用中可以使用模型来生成摘要
        summary = session_content[:500]  # 简化处理
        return self.add_memory(summary, memory_type="context")

    def get_all_memories(self, memory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有记忆"""
        if memory_type:
            return self.long_term_memory.get(memory_type, [])

        all_memories = []
        for m_type, memories in self.long_term_memory.items():
            all_memories.extend(memories)
        return all_memories

    def clear_memories(self, memory_type: Optional[str] = None):
        """清除记忆"""
        if memory_type:
            collection = self.collections.get(memory_type)
            if collection:
                # 清空集合
                import chromadb
                self.client.delete_collection(collection.name)
                self.collections[memory_type] = self.client.get_or_create_collection(
                    name=f"novel_{memory_type}",
                )
            self.long_term_memory[memory_type] = []
        else:
            # 清除所有记忆
            for m_type in self.MEMORY_TYPES:
                collection = self.collections.get(m_type)
                if collection:
                    self.client.delete_collection(collection.name)
                    self.collections[m_type] = self.client.get_or_create_collection(
                        name=f"novel_{m_type}",
                    )
            self.long_term_memory = {}

        self._save_long_term_memory()
        rprint(f"[yellow]已清除{memory_type or '所有'}记忆[/yellow]")

    def get_formatted_context(self, query: str) -> str:
        """获取格式化的上下文（用于生成）"""
        # 检索各类记忆
        context_parts = []

        # 人物记忆
        character_memory = self.get_character_memory()
        if character_memory:
            context_parts.append(character_memory)

        # 情节记忆
        plot_memory = self.get_plot_memory()
        if plot_memory:
            context_parts.append(plot_memory)

        # 环境设定
        setting_memories = self.retrieve_memory(query, memory_type="setting", top_k=2)
        if setting_memories:
            context_parts.append("【环境设定】")
            for mem in setting_memories:
                context_parts.append(f"- {mem['content']}")

        return "\n\n".join(context_parts)


# 全局单例
_manager: Optional[NovelMemoryManager] = None


def get_memory_manager() -> NovelMemoryManager:
    """获取记忆管理器单例"""
    global _manager
    if _manager is None:
        from config import config
        _manager = NovelMemoryManager(config)
        _manager.initialize()
    return _manager


# CLI测试
def main():
    """命令行测试"""
    from config import config

    manager = NovelMemoryManager(config)
    manager.initialize()

    # 测试添加记忆
    console.print("\n[bold cyan]测试添加记忆...[/bold cyan]")
    manager.add_character(
        name="李云",
        description="一位年轻的剑客，身世成谜",
        personality="冷静、坚毅",
        background="孤儿，被隐世高人收养",
    )
    manager.add_plot("李云在雪山偶遇受伤的老人，得知关于自己身世的线索")
    manager.add_setting("场景：终年积雪的天山之巅，云雾缭绕")

    # 测试检索记忆
    console.print("\n[bold cyan]测试检索记忆...[/bold cyan]")
    memories = manager.retrieve_memory("李云")
    for mem in memories:
        console.print(f"[green]{mem['type']}[/green]: {mem['content'][:50]}...")

    # 测试格式化上下文
    console.print("\n[bold cyan]测试格式化上下文...[/bold cyan]")
    context = manager.get_formatted_context("李云")
    console.print(context)


if __name__ == "__main__":
    main()
