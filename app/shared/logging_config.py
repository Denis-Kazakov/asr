import logging


def setup_logging(level):
    logging.basicConfig(
        level=level,
        # filename='app.log',
        # filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )