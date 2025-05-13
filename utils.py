import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'无效的日志级别: {log_level}')
    
    # 配置根日志记录器
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def format_timestamp(timestamp: float) -> str:
    """将时间戳格式化为可读的时间字符串"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL数据文件"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        logging.error(f"找不到文件: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"JSON解析错误: {file_path}")
        return []
    except Exception as e:
        logging.error(f"加载JSONL失败: {str(e)}")
        return []

def save_json(file_path: str, data: Any) -> bool:
    """保存数据为JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"保存JSON失败: {file_path}, 错误: {str(e)}")
        return False

def load_checkpoint(file_path: str) -> Optional[Dict[str, Any]]:
    """加载检查点数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"找不到检查点文件: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"检查点文件格式错误: {file_path}")
        return None
    except Exception as e:
        logging.error(f"加载检查点失败: {str(e)}")
        return None

def save_checkpoint(file_path: str, data: Dict[str, Any]) -> bool:
    """保存检查点数据"""
    return save_json(file_path, data)
