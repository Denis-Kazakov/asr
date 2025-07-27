import logging

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        # filename='app.log',
        # filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )