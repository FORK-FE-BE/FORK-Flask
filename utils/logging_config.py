import logging

def setup_logger():
    logger = logging.getLogger()
    if logger.hasHandlers():  # 이미 핸들러가 있으면 추가하지 않음
        return

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
