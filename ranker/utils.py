import re
from bs4 import BeautifulSoup
def name_processing(s: str):
    pattern1 = r"\((.*?)\)"
    s_new = re.sub(pattern1, "", s)
    pattern2 = r"\s+"
    result = re.sub(pattern2, " ", s_new)
    result = result.strip()
    result = [i.strip() for i in re.split(r'/|//|\\|\\\\', result) if len(i.strip())>0]
    if len(result)>0:
        return result[0]

def desc_processing(s: str):
    return BeautifulSoup(s).get_text()
