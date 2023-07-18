import azure.functions as func
import tiktoken
from opencensus.ext.azure.log_exporter import AzureEventHandler
import json
import logging
import os
import re
import math

conn_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
eventLogger = logging.getLogger(__name__)
eventLogger.addHandler(AzureEventHandler(connection_string=conn_string))
eventLogger.setLevel(logging.INFO)
# blueprint is a class that's instantiated to register functions outside of the core function application
bp = func.Blueprint() 
tokenizer = tiktoken.get_encoding("cl100k_base")
types_map = {
    dict.__name__: 'object',
    list.__name__: 'array',
    str.__name__: 'string',
    int.__name__: 'number',
    float.__name__: 'number',
    type(None).__name__: 'null',
}
@bp.function_name(name="CalculateTokens")
@bp.route(route="CalculateTokens")
def CalculateTokens(req: func.HttpRequest) -> func.HttpResponse:
    body = req.get_body()
    json_arr: list = json.loads(body)

    if type(json_arr) != list:
        return func.HttpResponse("Expecting array, got " + types_map[type(json_arr).__name__], status_code=422)

    if len(json_arr) == 0:
        return func.HttpResponse('[]', headers={'Content-Type': 'application/json'})


    lengths: list[int] = []
    for txt in json_arr:
        if type(txt) != str:
            return func.HttpResponse("Expecting array of strings, got " + types_map[type(txt).__name__], status_code=422)

        lengths.append(len(tokenizer.encode(txt)))

    return func.HttpResponse(json.dumps(lengths, ensure_ascii=False), headers={'Content-Type': 'application/json;charset=utf-8'})