from typing import TypedDict, NotRequired,Protocol,List


class Extras(TypedDict):
    signature: str               



class AIResponseBlock(TypedDict):
    type:    str                     
    text:    str                    
    extras:  NotRequired[Extras]   


class AIResponse(Protocol):

    def  __init__(self,content:List[AIResponseBlock]) -> None:
        self.content = content