import logging

# ________ set up logging ________
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
log = logging.getLogger(__name__)
f_handler = logging.FileHandler(f'{__name__}.log')
f_handler.setLevel(logging.INFO)
log.addHandler(f_handler)