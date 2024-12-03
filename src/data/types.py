from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, List, NewType, Optional, TypeVar

JSONSerializable = NewType("JSONSerializable", object)

if TYPE_CHECKING:
    from src.tasks.base import Task

_TaskT = TypeVar("_TaskT", bound="Task")


@dataclass
class CompanyDocument:
    """Dataclass for defining the company document in a structured fashion"""

    cvr: int
    sentences: List[List[str]]
    abspos: List[int]
    age: List[float]
    timecut_pos: int  
    segment: Optional[List[int]] = None
    background: Optional["Background"] = None
    shuffled: bool = False
    task_info: Optional[JSONSerializable] = None


@dataclass
class Background:
    """Defines the background information about a company
    TODO: Define the background sentence of a company. Until that, just return a single origin token which is DK in all cases. 
    """

    origin: str
    #OBS: Changes here must also be enforced in tasks.base.py at line 180 ish

    @staticmethod
    def get_sentence(x: Optional["Background"]) -> List[str]:
        """Return sequence of tokens corresponding to this person. Implemented as
        classmethod since we can null the background in PersonDocument in case of
        unknown background.
        """


        return ["[UNK]"]


class EncodedDocument(Generic[_TaskT]):
    """Generic class for encoded documents. Each task can then type-hint their
    specific encoding using a dataclass like

    .. code-block ::

        class MyTask:
            def encode_document(x: CompanyDocument) -> "MyTaskEncodedDocument":
                return MyTaskEncodedDocument(target=1)

        @dataclass
        class MyTaskEncodedDocument(EncodedDocument[MyTask]):
            target: int

    """
