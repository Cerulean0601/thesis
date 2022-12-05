import re

def preprocessingText(text):
    '''
        把標點符號和'\n'替換成空白，並且全部變為小寫
    '''
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text).lower()
    return text.replace('\n', ' ')