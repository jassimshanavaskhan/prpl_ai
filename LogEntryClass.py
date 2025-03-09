from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Tuple

# @dataclass
# class LogEntry:
#     timestamp: datetime
#     component: str
#     message: str
#     function_name: str = ""
#     struct_names: Set[str] = None
#     api_calls: Set[str] = None
    
#     def __post_init__(self):
#         self.struct_names = set() if self.struct_names is None else self.struct_names
#         self.api_calls = set() if self.api_calls is None else self.api_calls

@dataclass
class LogEntry:
    timestamp: datetime
    module: str
    level: str
    thread_id: str
    message: str
    component: str = ''
    function_name: str = ''
    api_calls: Set[str] = field(default_factory=set)