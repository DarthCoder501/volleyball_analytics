from loguru import logger


class ProjectLogger:
    def __init__(self, filename: str = "logs.log"):
        self.logger = logger
        self.logger.add(sink=filename, format="{time:MMMM D, YYYY > HH:mm:ss!UTC} | {level} | {message}",
                        serialize=True)
        # if show:
        #     self.logger.add(sink=sys.stderr, format="{time:MMMM D, YYYY > HH:mm:ss!UTC} | {level} | {message}")

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def success(self, msg):
        self.logger.success(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
