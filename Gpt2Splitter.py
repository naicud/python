import azure.functions as func
import tiktoken
from opencensus.ext.azure.log_exporter import AzureEventHandler
import json
import logging
import os
import re
import math
# blueprint is a class that's instantiated to register functions outside of the core function application

bp = func.Blueprint() 
conn_string = os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
eventLogger = logging.getLogger(__name__)
eventLogger.addHandler(AzureEventHandler(connection_string=conn_string))
eventLogger.setLevel(logging.INFO)

tokenizer = tiktoken.get_encoding("cl100k_base")
types_map = {
    dict.__name__: 'object',
    list.__name__: 'array',
    str.__name__: 'string',
    int.__name__: 'number',
    float.__name__: 'number',
    type(None).__name__: 'null',
}
@bp.function_name(name="Gpt2Splitter")
@bp.route(route="Gpt2Splitter")
def Gpt2Splitter(req: func.HttpRequest) -> func.HttpResponse:
    body = req.get_body()
    json_object: dict = json.loads(body)

    if type(json_object) != dict:
        return func.HttpResponse("Expecting object, got " + types_map[type(json_object).__name__], status_code=400)

    keys = json_object.keys()
    missing_properties = list(filter(lambda x: x not in keys, ['max_length', 'text']))
    if len(missing_properties) > 0:
        error_msg = f'Missing required {"property" if len(missing_properties) == 1 else "properties"} in body: {", ".join(missing_properties)}'
        return func.HttpResponse(error_msg, status_code=400)


    max_length = int(json_object['max_length'])
    full_text = str(json_object['text'])

    eventLogger.info('TokenizeText', extra={
        'custom_dimensions': {
            'FullText': (full_text[:252] + '...') if len(full_text) > 252 else full_text,
            'MaxLength': str(max_length)
        }
    })

    sentences = split_sentences(full_text)
    (split_text, split_tokens) = get_chunks(sentences, max_length)

    response_body = {
        'split_text': split_text,
        'split_tokens': split_tokens
    }

    return func.HttpResponse(
        json.dumps(response_body, ensure_ascii=False).encode('utf8'),
        headers={
            'Content-Type': 'application/json;charset=utf-8'
        }
    )


def split_sentences(text: str)-> list[str]:
    abbreviations = ['prot.', 'os.', 'prs.', 're.', 'lett.', 'abbr.', 'cfr.', 'art.', 'ss.mm.ii.']
    splitter = re.compile(r'(?<![\s.][A-Za-z0-9])\.[\s\n]+')

    candidate_sentences: list[str] = []
    for sentence in splitter.split(text):
        for s in re.split(r'(?:\r?\n)+', sentence):
            if not bool(s):
                continue
            candidate_sentences.append(s)

    sentences = []
    iter_candidates = iter(candidate_sentences)

    for sentence in iter_candidates:
        last_word = re.sub(r'^[^A-Za-z0-9]', '', sentence.split(' ')[-1])
        if last_word in abbreviations:
            next_sentence = next(iter_candidates)
            sentence = sentence + ' ' + next_sentence
        sentences.append(sentence)

    return sentences



def get_chunks(sentences: list[str], max_length: int):
    split_text = []
    split_tokens = []
    tokens_count: int = 0
    chunk_sentences: list[str] = []

    for sentence in sentences:
        tokens = tokenizer.encode(sentence)
        if tokens_count + len(tokens) > max_length:
            split_text.append(' '.join(chunk_sentences))
            split_tokens.append(tokens_count)
            tokens_count = 0
            chunk_sentences = []
        chunk_sentences.append(sentence)
        tokens_count = tokens_count + len(tokens)


    if tokens_count > 0:
        if tokens_count <= max_length * 0.8:
            fill_sentences = []
            sentences.reverse()

            if len(split_text) > 0:
                sentences = sentences[len(chunk_sentences):]
            else:
                iterations = math.ceil(max_length / tokens_count)
                sentences = sentences * iterations

            for sentence in sentences:
                tokens = tokenizer.encode(sentence)
                if tokens_count + len(tokens) > max_length:
                    break
                fill_sentences.append(sentence)
                tokens_count = tokens_count + len(tokens)

            chunk_sentences = fill_sentences + chunk_sentences


        split_text.append(' '.join(chunk_sentences))
        split_tokens.append(tokens_count)

    texts = []
    tokens = []
    for s, t in zip(split_text, split_tokens):
        if not bool(s):
            continue
        texts.append(s)
        tokens.append(t)

    return (texts, tokens)