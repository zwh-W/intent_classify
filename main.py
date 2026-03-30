# main.py
import time
import traceback
import uvicorn
from fastapi import FastAPI

# 导入数据格式规范
from data_schema import TextClassifyRequest, TextClassifyResponse
from logger import logger

# 导入我们的 4 大核心能力！
from model.regex_rule import model_for_regex
from model.tfidf_ml import model_for_tfidf
from model.bert import model_for_bert
from model.prompt import model_for_gpt

app = FastAPI(title="意图识别智能引擎 API", description="支持正则、TF-IDF、BERT、LLM-RAG 四种分类引擎")


# =====================================================================
# 内部通用处理器 (提取公共逻辑，体现高级工程师的 DRY 原则: Don't Repeat Yourself)
# =====================================================================
def process_classification(req: TextClassifyRequest, model_func, engine_name: str) -> TextClassifyResponse:
    start_time = time.time()
    logger.info(f"[{engine_name} 引擎] 收到请求 ID: {req.request_id} | 文本: {req.request_text}")

    result = ""
    error_msg = "ok"
    try:
        # 调用传进来的模型函数进行推理
        result = model_func(req.request_text)
    except Exception as e:
        logger.error(f"[{engine_name} 引擎] 推理报错: {e}")
        error_msg = traceback.format_exc()

    cost_time = round(time.time() - start_time, 3)
    logger.info(f"[{engine_name} 引擎] 预测结果: {result} | 耗时: {cost_time}s")

    return TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result=result,
        classify_time=cost_time,
        error_msg=error_msg
    )


# =====================================================================
# 暴露 4 个对外的 API 接口
# =====================================================================

@app.post("/v1/text-cls/regex", response_model=TextClassifyResponse, summary="1. 正则规则引擎")
def regex_classify(req: TextClassifyRequest):
    return process_classification(req, model_for_regex, "Regex")


@app.post("/v1/text-cls/tfidf", response_model=TextClassifyResponse, summary="2. TF-IDF 机器学习引擎")
def tfidf_classify(req: TextClassifyRequest):
    return process_classification(req, model_for_tfidf, "TF-IDF")


@app.post("/v1/text-cls/bert", response_model=TextClassifyResponse, summary="3. BERT 深度学习引擎")
def bert_classify(req: TextClassifyRequest):
    return process_classification(req, model_for_bert, "BERT")


@app.post("/v1/text-cls/gpt", response_model=TextClassifyResponse, summary="4. LLM 大模型 RAG 引擎")
def gpt_classify(req: TextClassifyRequest):
    return process_classification(req, model_for_gpt, "LLM-RAG")


# =====================================================================
# 服务启动入口
# =====================================================================
if __name__ == "__main__":
    logger.info("准备启动 FastAPI 意图识别服务...")
    # uvicorn 是跑 FastAPI 的高性能服务器
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)