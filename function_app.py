import azure.functions as func 
from CalculateTokens import bp as CalculateTokensBluePrint
from Gpt2Splitter import bp as Gpt2SplitterBluePrint
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION) 

app.register_functions(CalculateTokensBluePrint) 
app.register_functions(Gpt2SplitterBluePrint) 