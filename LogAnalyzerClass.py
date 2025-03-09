from typing import Dict, List, Set, Any, Optional, Tuple
import re
from logger import logger
from datetime import datetime, timedelta
from pathlib import Path
from LogEntryClass import LogEntry

class LogAnalyzer:
    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)
        # Updated pattern for new log format
        self.log_pattern = re.compile(
            r'(?P<timestamp>\d{6}-\d{2}:\d{2}:\d{2}\.\d{6})\s+'  # Timestamp
            r'\[mod=(?P<module>\w+),\s*lvl=(?P<level>\w+)\]\s+'  # Module and Level
            r'\[tid=(?P<thread_id>\d+)\]\s+'                     # Thread ID
            r'(?P<message>.*?)$'                                 # Rest of the message
        )
        
        # Patterns for specific component messages
        self.component_patterns = {
            'rbus': re.compile(r'rbus.*?:\s*(.*?)(?=\]|\[|$)', re.IGNORECASE),
            'ccsp': re.compile(r'CCSP.*?:\s*(.*?)(?=\]|\[|$)', re.IGNORECASE),
            'cr': re.compile(r'CR\s+(?:INFO|WARN|ERROR)\s+(.*?)(?=\]|\[|$)', re.IGNORECASE)
        }
        
        self.function_pattern = re.compile(r'(?:Entering|Exiting|Called|Executing)\s+(\w+)')
        self.api_pattern = re.compile(r'(CCSP_|RDK_|RBUS_|TR181_|CcspCommon_|DM_|PSM_)\w+')
        
    def parse_log_files(self) -> List[LogEntry]:
        """Parse all log files in the directory with enhanced format parsing"""
        log_entries = []
        
        for log_file in self.log_directory.glob('*.txt*'):
            try:
                entries = self._parse_single_log(log_file)
                log_entries.extend(entries)
            except Exception as e:
                print(f"Error parsing log file {log_file}: {str(e)}")
                
        return sorted(log_entries, key=lambda x: x.timestamp)
    
    def _parse_single_log(self, log_file: Path) -> List[LogEntry]:
        """Parse a single log file with the new format"""
        entries = []
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    match = self.log_pattern.match(line.strip())
                    if not match:
                        continue
                        
                    # Extract basic information
                    timestamp = self._parse_timestamp(match.group('timestamp'))
                    module = match.group('module')
                    level = match.group('level')
                    thread_id = match.group('thread_id')
                    message = match.group('message')
                    
                    # Create log entry
                    entry = LogEntry(
                        timestamp=timestamp,
                        module=module,
                        level=level,
                        thread_id=thread_id,
                        message=message,
                        component=self._determine_component(message, module)
                    )
                    
                    # Extract additional information
                    self._extract_function_info(message, entry)
                    self._extract_api_calls(message, entry)
                    entries.append(entry)
                    
                except Exception as e:
                    print(f"Error parsing line: {str(e)}")
                    continue
                    
        return entries
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from new format: YYMMDD-HH:MM:SS.NNNNNN"""
        try:
            # Add '20' prefix for year since the format uses 2-digit year
            year = f"20{timestamp_str[:2]}"
            month = timestamp_str[2:4]
            day = timestamp_str[4:6]
            time = timestamp_str[7:]
            
            full_timestamp = f"{year}-{month}-{day} {time}"
            return datetime.strptime(full_timestamp, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError as e:
            print(f"Error parsing timestamp {timestamp_str}: {str(e)}")
            return datetime.now()
    
    def _determine_component(self, message: str, module: str) -> str:
        """Determine component from message content and module"""
        # First check if it's a known component based on the module
        module_component_map = {
            'CR': 'CcspCr',
            'PSM': 'CcspPsm',
            'WANMGR': 'RdkWanManager',
            'WIFI': 'RdkWifiManager',
            'TR69': 'CcspTr069Pa',
            'ETHAGT': 'CcspEthAgent',
            'WEBPA': 'webpa'
        }
        
        if module in module_component_map:
            return module_component_map[module]
            
        # Check message content for component information
        if 'com.cisco.spvtg.ccsp.' in message:
            component = message.split('com.cisco.spvtg.ccsp.')[1].split()[0]
            return f"Ccsp{component}"
            
        return module  # Default to module name if no specific component found
    
    def _extract_function_info(self, message: str, entry: LogEntry):
        """Extract function name from log message"""
        func_match = self.function_pattern.search(message)
        if func_match:
            entry.function_name = func_match.group(1)
            
        # Additional function name extraction for specific formats
        if '.c:' in message and '--' in message:
            try:
                file_func = message.split('.c:')[0].split()[-1]
                entry.function_name = file_func
            except:
                pass

    def _extract_api_calls(self, message: str, entry: LogEntry):
        """Extract API calls from log message"""
        api_calls = set(self.api_pattern.findall(message))
        if api_calls:
            entry.api_calls.update(api_calls)