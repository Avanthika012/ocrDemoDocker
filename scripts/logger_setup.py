import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

load_dotenv()  # loading env variables

class ColoredFormatter(logging.Formatter):
    """ Custom formatter to add colors to log levels """
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Fore.RESET}"
        return super().format(record)

class InitLoggers:
    """
    - can log info, error
    - can traceback if any error occurs 
    - if console_output=True --> all logs can be printed on terminal or console
    - if console_output=False --> all logs will be stored in the log file. All error msgs will be stored on logs files and will be displayed on terminal
    """
    def __init__(self, name, max_bytes, backup_count, logfile_name="preprocess.log", save_path="logs/", console_output=True):
        # Setting formatter
        self.formatter = ColoredFormatter(
            '%(asctime)s [%(levelname)8s:%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        
        # setting up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # won't show output on console
        os.makedirs(save_path, exist_ok=True)
        
        # File handler
        file_handler = RotatingFileHandler(os.path.join(save_path, logfile_name),
                                           maxBytes=self.max_bytes,
                                           backupCount=self.backup_count)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        if console_output:
            # Console handler for all logs
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)
        else:
            # Error handler for error messages if console output is False
            error_handler = logging.StreamHandler()
            error_handler.setFormatter(self.formatter)
            error_handler.setLevel(logging.ERROR)
            self.logger.addHandler(error_handler)
            
def get_logger(name, logfile_name, console_output=True):
    max_bytes = os.getenv("MAXBYTES_LOGGER", 1048576)  # default 1 MB if not set
    backup_count = os.getenv("BACKUPCOUNT_LOGGER", 5)  # default 5 if not set
    logger_instance = InitLoggers(name, max_bytes, backup_count, logfile_name=logfile_name, save_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs'), console_output=console_output).logger
    return logger_instance
