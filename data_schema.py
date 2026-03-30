# data_schema.py
from pydantic import BaseModel, Field
from typing import Union, List, Optional

class TextClassifyRequest(BaseModel):
    """
    接收请求的格式 (用户的点菜单)
    """
    request_id: Optional[str] = Field(default="test_001", description="请求id, 方便追踪日志")
    request_text: Union[str, List[str]] = Field(..., description="要分类的文本，支持单句话(str)或批量(List[str])")

class TextClassifyResponse(BaseModel):
    """
    返回给用户的格式 (给用户上的菜)
    """
    request_id: Optional[str] = Field(..., description="原样返回请求id")
    request_text: Union[str, List[str]] = Field(..., description="原样返回请求的文本")
    classify_result: Union[str, List[str]] = Field(..., description="模型预测的意图类别")
    classify_time: float = Field(..., description="模型推理耗时(秒)")
    error_msg: str = Field(default="ok", description="异常信息，正常则为ok")